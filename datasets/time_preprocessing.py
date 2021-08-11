# Optional script for pre-processing original dataset with an option for pre-processing timestamp data
# Timestamp modelling can help improve the performance for RecSys


import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from tqdm import tqdm


class opt:
    dataset = "yoochoose"


dataset = 'train-item-views.csv'

if opt.dataset == 'yoochoose':
    dataset = 'yoochoose/yoochoose-clicks.dat'


print("-- Starting @ %ss" % datetime.datetime.now())

try:
    sess_clicks, sess_date = pickle.load(
        open("./yoochoose_full/temp.pkl", "rb"))
    print("Loaded saved intermediate pickle")

except:
    with open(dataset, "r") as f:
        if opt.dataset == 'yoochoose':
            reader = csv.DictReader(f, delimiter=',')
        else:
            reader = csv.DictReader(f, delimiter=';')
        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for data in tqdm(reader):
            sessid = data['session_id']
            if curdate and not curid == sessid:
                date = ''
                if opt.dataset == 'yoochoose':
                    date = time.mktime(time.strptime(
                        curdate[:19], '%Y-%m-%dT%H:%M:%S'))
                else:
                    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid
            if opt.dataset == 'yoochoose':
                item, cat = data['item_id'], data['category_id']
            else:
                item = data['item_id'], int(data['timeframe'])
            curdate = ''
            if opt.dataset == 'yoochoose':
                curdate = data['timestamp']
            else:
                curdate = data['eventdate']

            temp_curdate = datetime.datetime.strptime(
                curdate, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
            if sessid in sess_clicks:
                sess_clicks[sessid] += [(item, cat, temp_curdate)]
            else:
                sess_clicks[sessid] = [(item, cat, temp_curdate)]

        date = ''
        if opt.dataset == 'yoochoose':
            date = time.mktime(time.strptime(
                curdate[:19], '%Y-%m-%dT%H:%M:%S'))
        else:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            for i in list(sess_clicks):
                sorted_clicks = sorted(
                    sess_clicks[i], key=operator.itemgetter(1))
                sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_date[curid] = date

    pickle.dump((sess_clicks, sess_date), open(
        "./yoochoose_full/temp.pkl", "wb"))
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid, cat, timestamp in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i[0]] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1
else:
    splitdate = maxdate - 86400 * 7

# Yoochoose: ('Split date', 1411930799.0)
print('Splitting date', splitdate)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
# [(session_id, timestamp), (), ]
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
# [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())
print("Session: ", sess_clicks[tra_sess[0][0]])

# from numba import njit
# @njit(parallel=True)


def delete_dups(a, b, c):
    last_item = a[0]
    to_keep = [0]
    for i in range(1, len(a)):
        if a[i] != last_item:
            last_item = a[i]
            to_keep.append(i)

    new_a = []
    new_b = []
    new_c = []

    for i in range(len(a)):
        if i in to_keep:
            new_a.append(a[i])
            new_b.append(b[i])
            new_c.append(c[i])

    # assert(len(new_a) == len(new_b) == len(new_c))
    return new_a, new_b, new_c


# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
cat_dict = {}
# Convert training sessions to sequences and renumber items to start from 1


def obtain_tra():
    train_sessid = []
    train_cats = []
    train_i = []
    train_timestamp = []
    train_dates = []

    item_ctr = 1
    cat_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq_i = []
        outseq_cat = []
        outseq_timestamp = []

        for i, cat, timestamp in seq:
            if i not in item_dict:
                item_dict[i] = item_ctr
                item_ctr += 1
            if cat not in cat_dict:
                cat_dict[cat] = cat_ctr
                cat_ctr += 1
            outseq_i += [item_dict[i]]
            outseq_cat += [cat_dict[cat]]
            outseq_timestamp += [timestamp]
        if len(outseq_i) < 2:  # Doesn't occur
            continue
        train_sessid += [s]
        train_dates += [date]
        train_i += [outseq_i]
        train_cats += [outseq_cat]
        train_timestamp += [outseq_timestamp]
    print(item_ctr)     # 43098, 37484

    for sess_id in range(len(train_i)):
        train_i[sess_id], train_cats[sess_id], train_timestamp[sess_id] = delete_dups(
            train_i[sess_id], train_cats[sess_id], train_timestamp[sess_id])
    return train_sessid, train_dates, train_i, train_cats, train_timestamp


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtain_tes():
    test_sessid = []
    test_i = []
    test_cat = []
    test_timestamp = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq_i = []
        outseq_cat = []
        outseq_timestamp = []
        for i, cat, timestamp in seq:
            if i in item_dict:
                outseq_i += [item_dict[i]]
                outseq_cat += [cat]
                outseq_timestamp += [timestamp]
        if len(outseq_i) < 2:
            continue
        test_sessid += [s]
        test_dates += [date]
        test_i += [outseq_i]
        test_cat += [outseq_cat]
        test_timestamp += [outseq_timestamp]

    for sess_id in range(len(test_i)):
        test_i[sess_id], test_cat[sess_id], test_timestamp[sess_id] = delete_dups(
            test_i[sess_id], test_cat[sess_id], test_timestamp[sess_id])

    return test_sessid, test_dates, test_i, test_cat, test_timestamp


def process_seqs(iseqs, idates, category_ids, timestamps):
    out_seqs = []
    out_dates = []
    out_cats = []
    out_timestamps = []
    labs = []
    ids = []

    for id, seq, date, cat, ts in zip(tqdm(range(len(iseqs))), iseqs, idates, category_ids, timestamps):
        tar = seq[1:]
        labs += [tar]
        out_seqs += [seq[:-1]]
        out_dates += [date]
        out_cats += [cat[:-1]]
        out_timestamps += [ts[:-1]]
        ids += [id]
    return out_seqs, out_cats, out_timestamps, out_dates, labs, ids


print("train start")
train_sessid, train_dates, train_i, train_cats, train_timestamp = obtain_tra()
print("train done 1/2")
tr_seqs, tr_cats, tr_timestamps, tr_dates, tr_labs, tr_ids = process_seqs(
    train_i, train_dates, train_cats, train_timestamp)
print("tr_timestamps : ", tr_timestamps[:30])

time_stamps = []
for idx, item in enumerate(tr_timestamps):
    new_item = []
    if len(item) == 1:
        new_item.append(0)
    else:
        new_item.append(0)
        for ix, clk in enumerate(item[:-1]):
            new_item.append(item[ix+1] - item[ix])

    time_stamps.append(new_item)
print("tr_timestamps_new : ", time_stamps[:30])
tr_timestamps = time_stamps
del time_stamps
print("train done 2/2")
tra = (tr_seqs, tr_cats, tr_timestamps, tr_labs)
del tr_seqs, tr_cats, tr_timestamps, tr_dates, tr_labs, tr_ids
del train_sessid, train_dates, train_i, train_cats, train_timestamp
del tra_sess

if not os.path.exists('diginetica'):
    os.makedirs('diginetica')
if not os.path.exists('yoochoose_full'):
    os.makedirs('yoochoose_full')

if opt.dataset == 'diginetica':
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
else:
    pickle.dump(tra, open('yoochoose_full/train.pkl', 'wb'))
print("train dumped")

test_sessid, test_dates, test_i, test_cats, test_timestamp = obtain_tes()
te_seqs, te_cats, te_timestamps, te_dates, te_labs, te_ids = process_seqs(
    test_i, test_dates, test_cats, test_timestamp)

print("te_timestamps : ", te_timestamps)

te_time_stamps = []
for idx, item in enumerate(te_timestamps[:30]):
    new_item = []
    if len(item) == 1:
        new_item.append(0)
    else:
        new_item.append(0)
        for ix, clk in enumerate(item[:-1]):
            new_item.append(item[ix+1] - item[ix])

    te_time_stamps.append(new_item)

print("te_timestamps_new : ", te_time_stamps[:30])
te_timestamps = te_time_stamps
del te_time_stamps
tes = (te_seqs, te_cats, te_timestamps, te_labs)
del te_seqs, te_cats, te_timestamps, te_dates, te_labs, te_ids
del test_sessid, test_dates, test_i, test_cats, test_timestamp
del sess_clicks, sess_date
del tes_sess

if opt.dataset == 'diginetica':
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
else:
    pickle.dump(tes, open('yoochoose_full/test.pkl', 'wb'))


print('Done')

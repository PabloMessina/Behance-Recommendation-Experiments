def bar_chart(ax, col_heights, col_labels, xlabel, ylabel, title,
              vertical=True,
              x_rot=0, y_rot=0,
              fontsize_title=15, fontsize_axis_label=15, fontsize_col_text=15,
              fontsize_xticks=12, fontsize_yticks=12,
              title_ha='center', title_position=None,
              xlim=None, ylim=None,
              col_texts=None):
    """
    generic function to plot vertical/horizontal barcharts
    """

    import matplotlib.pyplot as plt
    import numpy as np
    
    # set ticks configuration
    plt.xticks(rotation=x_rot, fontsize=fontsize_xticks)
    plt.yticks(rotation=y_rot, fontsize=fontsize_yticks)

    # generate bars and labels
    n = len(col_heights)
    ind = np.arange(n)

    if vertical:
        ax.bar(ind + 0.75, col_heights, 0.5, color='b')
        ax.set_xticks(ind + 0.75)
        ax.set_xticklabels(col_labels, ha="center")
    else: # horizontal
        ax.barh(ind + 0.75, col_heights, 0.5, color='b')
        ax.set_yticks(ind + 0.75)
        ax.set_yticklabels(col_labels, va="center")

    ax.set_xlabel(xlabel, fontsize=fontsize_axis_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis_label)
    ax.set_title(title, fontsize=fontsize_title, ha=title_ha)
    if title_position is not None:
        ax.title.set_position(title_position)

    # set display limits
    if vertical:
        xlim = xlim or [0, n+0.5]
        ylim = ylim or [0, max(col_heights) * 1.45]
    else:
        xlim = xlim or [0, max(col_heights) * 1.45]
        ylim = ylim or [0, n+0.5]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # add texts next to columns
    if col_texts is not None:

        rects = ax.patches
        delta = max(col_heights) * 0.03

        if vertical:
            for rect, txt in zip(rects, col_texts):
                ax.text(rect.get_x() + rect.get_width() * 0.5, rect.get_height() + delta,
                        txt, ha='center', va='bottom', fontsize=fontsize_col_text)
        else:
            for rect, txt in zip(rects, col_texts):
                ax.text(rect.get_width() + delta, rect.get_y() + rect.get_height() * 0.5,
                        txt, ha='left', va='center', fontsize=fontsize_col_text)

def long_tail_barchart(numbers, threshold, xlabel, ylabel, title, figsize=None, **kwargs):
    
    count_map = dict()
    for num in numbers:
        if num in count_map:
            count_map[num]+=1
        else:
            count_map[num]=1
    count_pairs = [(k,v) for k,v in count_map.items()]
    count_pairs.sort(key=lambda p: p[0])
    values = [v for v,c in count_pairs]
    counts = [c for v,c in count_pairs]
    
    col_labels = []
    col_heights = []
    tot = sum(counts)
    acc = 0
    for v,c in zip(values, counts):
        if acc / tot >= threshold:
            col_labels.append('{}+'.format(v))
            col_heights.append(tot - acc)
            break
        col_labels.append(str(v))
        col_heights.append(c)
        acc+=c
        
    col_texts = ["{:.2f}%".format(float(h/tot*100)) for h in col_heights]
    n = len(col_labels)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    bar_chart(ax, col_heights, col_labels, xlabel, ylabel, title, col_texts=col_texts, vertical=True, **kwargs)    

def load_behance_features(path):
    import struct
    import numpy as np
    from os.path import getsize

    filesize = getsize(path)
    n_items = filesize // (8 + 4*4096)
    print ('bytes:', filesize)
    print('n_items:', n_items)
    
    item_ids = np.empty((n_items,), dtype=int)
    featmat = np.empty((n_items, 4096), dtype=float)
    with open(path, 'rb') as f:
        for i in range(n_items):
            item_ids[i] = int(f.read(8))
            featmat[i] = struct.unpack('f'*4096, f.read(4*4096))
    item_id2index = {_id:i for i,_id in enumerate(item_ids)}
    return dict(
        ids=item_ids,
        id2index=item_id2index,
        featmat=featmat,
    )

def load_behance_item_ids_with_features(path):
    import numpy as np
    from os.path import getsize

    filesize = getsize(path)
    n_items = filesize // (8 + 4*4096)
    print ('bytes:', filesize)
    print('n_items:', n_items)
    
    item_ids = np.empty((n_items,), dtype=int)
    with open(path, 'rb') as f:
        for i in range(n_items):
            item_ids[i] = int(f.read(8))
            f.seek(4*4096, 1)
    item_id2index = {_id:i for i,_id in enumerate(item_ids)}
    assert len(item_ids) == len(item_id2index)
    assert len(item_ids) == n_items
    return dict(
        ids=item_ids,
        id2index=item_id2index,
    )

def pairs2dict(pairs):
    tmp = dict()
    for x,y in pairs:
        try:
            tmp[x].append(y)
        except KeyError:
            tmp[x] = [y]
    return tmp

def tuples2dict(tuples):
    tmp = dict()
    for t in tuples:
        x = t[0]
        y = t[1:]
        try:
            tmp[x].append(y)
        except KeyError:
            tmp[x] = [y]
    return tmp

def AUC(relevant_positions, inventory_size):
    n = len(relevant_positions)
    assert inventory_size >= n
    if inventory_size == n:
        return 1
    auc = 0
    for i, idx in enumerate(relevant_positions):
        auc += ((inventory_size - (idx+1)) - (n - (i+1))) / (inventory_size - n)
    auc /= n
    return auc

def run_experiment(compute_AUC_func, save_dir_path, method_name):    
    assert save_dir_path[-1] == '/'
    import numpy as np
    from time import time
    
    start_t = time()
    
    # load train/test data
    train_array = np.load('/mnt/workspace/Behance/train.npy')
    test_pos_array = np.load('/mnt/workspace/Behance/test_pos.npy')
    test_neg_array = np.load('/mnt/workspace/Behance/test_neg.npy')
    test_users = np.load('/mnt/workspace/Behance/test_users.npy')
    
    # regroup data into dictionaries
    user2items_train = pairs2dict(train_array[:,:2])
    user2items_test_pos = pairs2dict(test_pos_array[:,:2])
    user2items_test_neg = pairs2dict(test_neg_array)
    
    # compute AUC for each test instance
    aucs = np.empty((len(test_users),), dtype=float)
    for j, u in enumerate(test_users):
        train_items = user2items_train[u]
        test_pos_items = user2items_test_pos[u]
        test_neg_items = user2items_test_neg[u]    
        aucs[j] = compute_AUC_func(train_items, test_pos_items, test_neg_items)
        
    # save results
    from os import makedirs
    makedirs(save_dir_path, exist_ok=True)
    output_path = '%s%s_aucs.npy' % (save_dir_path, method_name)
    np.save(output_path, aucs)
    print('experiment successfully finished: results saved to %s' % output_path)
    print('\t elapsed_seconds = %.2f, mean_AUC = %.5f' % (time() - start_t, aucs.mean()))

def run_experiment__timeaware(process_like_func, compute_AUC_func, save_dir_path, method_name):
    assert save_dir_path[-1] == '/'
    import numpy as np
    from time import time
    
    start_t = time()
    
    # load train/test data
    train_array = np.load('/mnt/workspace/Behance/train.npy')
    test_pos_array = np.load('/mnt/workspace/Behance/test_pos.npy')
    test_neg_array = np.load('/mnt/workspace/Behance/test_neg.npy')
    test_users = np.load('/mnt/workspace/Behance/test_users.npy')
    
    user2index = {u:i for i,u in enumerate(test_users)}
    user2pairs_test_pos = tuples2dict(test_pos_array)
    user2items_test_neg = pairs2dict(test_neg_array)
    
    # collect events
    events = []    
    for u, i, t in train_array:
        events.append((t,u,i,0))
    for u, pairs in user2pairs_test_pos.items():
        assert all(pairs[i][1] == pairs[i-1][1] for i in range(len(pairs)))
        t = pairs[0][1]
        items = [p[0] for p in pairs]
        events.append((t,u,items,1))
    events.sort(key=lambda e:e[0])
    print('len(events) = ', len(events))
    
    # compute AUC for each test instance
    n_test_users = len(test_users)
    tested_users = set()
    aucs = np.full((n_test_users,),-1, dtype=float)
    for event in events:
        etype = event[3]
        if etype == 0: # train
            t,u,i,_ = event
            assert type(i) is np.int64
            process_like_func(u,i,t)
        else: # test pos
            t,u,pos_items,_ = event
            assert etype == 1
            assert type(pos_items) is list
            assert len(pos_items) > 0
            neg_items = user2items_test_neg[u]            
            aucs[user2index[u]] = compute_AUC_func(u, pos_items, neg_items, t)
            tested_users.add(u)
    assert aucs.min() >= 0
    assert len(tested_users) == n_test_users
        
    # save results
    from os import makedirs
    makedirs(save_dir_path, exist_ok=True)
    output_path = '%s%s_aucs.npy' % (save_dir_path, method_name)
    np.save(output_path, aucs)
    print('experiment successfully finished: results saved to %s' % output_path)
    print('\t elapsed_seconds = %.2f, mean_AUC = %.5f' % (time() - start_t, aucs.mean()))
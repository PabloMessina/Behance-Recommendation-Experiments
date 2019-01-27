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
    
    f = open(path, 'rb')
    item_ids = np.empty((n_items,))
    featmat = np.empty((n_items, 4096))
    for i in range(n_items):
        item_ids[i] = int(f.read(8))
        featmat[i] = struct.unpack('f'*4096, f.read(4*4096))
    item_id2index = {_id:i for i,_id in enumerate(item_ids)}
    return dict(
        ids=item_ids,
        id2index=item_id2index,
        featmat=featmat,
    )
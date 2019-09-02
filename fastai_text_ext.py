from fastai.text import *


def show_validation_result(interp:TextClassificationInterpretation, k:int):
    """
     Return a dataframe showing the first `k` texts in validation along with their prediction, actual class.
    """
    items = []
    tl_val ,tl_idx = interp.top_losses()

    for i ,idx in enumerate(tl_idx):
        if k <= 0; break
        k -= 1
        tx ,cl = interp.data.dl(interp.ds_type).dataset[idx]
        cl = cl.data
        classes = interp.data.classes
        txt = tx.text
        tmp = [txt, f'{classes[interp.pred_class[idx]]}', f'{classes[cl]}']
        items.append(tmp)

    items = np.array(items)
    names = ['Text', 'Prediction', 'Actual']
    df = pd.DataFrame({n :items[: ,i] for i ,n in enumerate(names)}, columns=names)

    return df
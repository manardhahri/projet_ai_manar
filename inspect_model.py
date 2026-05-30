import h5py, json

with h5py.File('model/mon_modele_melanome_vgg16.h5', 'r') as f:
    cfg = json.loads(f.attrs['model_config'])
    layers = cfg['config']['layers']
    for l in layers:
        lc = l.get('config', {})
        cls = l.get('class_name', '?')
        if cls == 'Dense':
            print("Dense units=%s activation=%s" % (lc.get('units'), lc.get('activation')))
        elif cls == 'Dropout':
            print("Dropout rate=%s" % lc.get('rate'))
        elif cls == 'InputLayer':
            print("InputLayer batch_shape=%s" % (lc.get('batch_shape') or lc.get('batch_input_shape')))

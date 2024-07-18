from keras_unet_collection import models

def unet_3plus_model(input_size=(256, 256, 3), num_classes=8, backbone="ResNet50V2"):
    model = models.unet_3plus_2d(input_size, n_labels=num_classes, filter_num_down=[64, 128, 256, 512],
                                 filter_num_skip='auto', filter_num_aggregate='auto', stack_num_down=2, stack_num_up=2,
                                 activation='ReLU', output_activation='Softmax', batch_norm=True, pool=True,
                                 unpool=True, deep_supervision=False, backbone=backbone, weights='imagenet',
                                 freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')
    return model

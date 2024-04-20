DATA = {
    'data_root': "/SSD1/minhyeok/dataset/VOS",

    'pretrain': "DUTS_train",

    'best_pretrained_model': "./log/2022-11-05 19:59:34/model/best_model.pth",
    'DAVIS_train_main': "DAVIS_train",
    'DAVIS_train_sub': "YTVOS_train", # or None
    'DAVIS_val': "DAVIS_test",
    
    'best_model': "/SSD1/minhyeok/test/log/2022-11-07 13:39:05/model/best_model.pth",
    'FBMS_test': "FBMS_test",
    'YTobj_test': "YTobj_test",
}

TRAIN = {
    'GPU': "0, 1",
    'epoch': 200,
    'learning_rate': 1e-4,
    'print_freq': 50,
    'batch_size': 12,
    'img_size': 512
}
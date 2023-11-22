general_arguments = [
    'gpu_id', 'use_gpu',
    'seed',
    'reproducibility',
    'state',
    'checkpoint_dir',
    'show_progress',
    'config_file',
    'log_wandb',
]

training_arguments = [
    'epochs', 'train_batch_size',
    'learner', 'learning_rate',
    'eval_step', 'stopping_step',
    'weight_decay', 'resume'
    'train'
]

evaluation_arguments  = [
    'metrics',
    'temp_dir',
    'evaluate_batch_size',
    'lse_checkpoint_path',
    'valid_metric_bigger',
]
# evaluation_arguments = [
#     'eval_args', 'repeatable',
#     'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
#     'eval_batch_size',
#     'metric_decimal_place',
# ]

# dataset_arguments = [
#     'field_separator', 'seq_separator',
#     'USER_ID_FIELD', 'ITEM_ID_FIELD', 'RATING_FIELD', 'TIME_FIELD',
#     'seq_len',
#     'LABEL_FIELD', 'threshold',
#     'NEG_PREFIX',
#     'ITEM_LIST_LENGTH_FIELD', 'LIST_SUFFIX', 'MAX_ITEM_LIST_LENGTH', 'POSITION_FIELD',
#     'HEAD_ENTITY_ID_FIELD', 'TAIL_ENTITY_ID_FIELD', 'RELATION_ID_FIELD', 'ENTITY_ID_FIELD',
#     'load_col', 'unload_col', 'unused_col', 'additional_feat_suffix',
#     'rm_dup_inter', 'val_interval', 'filter_inter_by_user_or_item',
#     'user_inter_num_interval', 'item_inter_num_interval',
#     'alias_of_user_id', 'alias_of_item_id', 'alias_of_entity_id', 'alias_of_relation_id',
#     'preload_weight', 'normalize_field', 'normalize_all',
#     'benchmark_filename',
# ]

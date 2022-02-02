{
    WEIGHTS_NAME: "pytorch_model.bin",
    checkpoint: "../skl_model/",
    model_name_or_path: "../prev_trained_model/chinese_roberta_wwm_ext_pytorch",
    task_name: "sklJoint",
    device: "cpu",
    markup: "bio",
    eval_max_seq_length: 512,
    model_type: "bert",
    do_lower_case: true,
    batch_size: 16
}

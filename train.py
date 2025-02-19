import time
from transformers import (
    DefaultDataCollator,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataset import LLMDataset


def load_model(model_name):
    if model_name == "mini_llama":
        from model import Config, MiniLLAMA

        config = Config()
        model = MiniLLAMA(config)
        return model, config

    elif model_name == "mini_moe":
        from moe_model import Config, MiniMOE

        config = Config()
        model = MiniMOE(config)
        return model, config

    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":

    model_name = "mini_moe"
    time_stamp = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    output_dir = f"./results_{model_name}/pretrain/{time_stamp}"

    model, config = load_model(model_name)
    config.output_dir = output_dir
    model.train()
    print("\n\n", "#" * 20, "Config", "#" * 20)
    for k, v in config.__dict__.items():
        print(k, v)
    print("#" * 20, "######", "#" * 20, "\n\n")

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model # param = {n_param}")

    tokenizer_path = ""
    data_path = ""
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    start_time = time.time()
    dataset = LLMDataset(data_path, tokenizer, max_seq_len=config.max_seq_len)

    print(f"Dataset ({len(dataset)} lines) loaded in {time.time() - start_time: .2f}s.")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        do_train=True,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to="tensorboard",
        save_total_limit=5,
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        save_safetensors=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(f"{output_dir}/final_model")
    trainer.save_state()

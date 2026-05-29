import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import MiniMindConfig, mokioMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = mokioMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind Pretrain Model Eval")

    parser.add_argument('--load_from', default='model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)

    # 改成pretrain
    parser.add_argument('--weight', default='pretrain', type=str)
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）",
    )
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)

    parser.add_argument('--use_moe', default=0, type=int)
    parser.add_argument('--inference_rope_scaling', action='store_true')

    parser.add_argument('--max_new_tokens', default=256, type=int)
    parser.add_argument('--temperature', default=0.8, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)

    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str
    )

    args = parser.parse_args()

    model, tokenizer = init_model(args)

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    prompts = [
        "中国的首都是",
        "请解释Transformer的工作原理：",
        "Python实现快速排序：",
        "今天天气很好，我们",
        "人工智能未来的发展方向包括",
    ]

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('Prompt >>> '), '')

    for prompt in prompt_iter:

        setup_seed(random.randint(0, 999999))

        print(f"\nPrompt:\n{prompt}")

        # pretrain模型只做纯文本续写
        text = tokenizer.bos_token + prompt

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(args.device)

        print("\nCompletion:\n", end='')

        st = time.time()

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],

                max_new_tokens=args.max_new_tokens,

                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,

                repetition_penalty=1.1,

                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,

                streamer=streamer
            )

        gen_tokens = (
            generated_ids.shape[1]
            - inputs["input_ids"].shape[1]
        )

        speed = gen_tokens / (time.time() - st)

        print(f"\n\n[Generated Tokens]: {gen_tokens}")
        print(f"[Speed]: {speed:.2f} tokens/s\n")

if __name__ == "__main__":
    main()
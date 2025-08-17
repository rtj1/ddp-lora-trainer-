import argparse
def main():
    ap = argparse.ArgumentParser(description="Compute scaling efficiency")
    ap.add_argument("--tokens1", type=float, required=True, help="Tokens/sec on 1 GPU")
    ap.add_argument("--tokensN", type=float, required=True, help="Tokens/sec on N GPUs")
    ap.add_argument("--N", type=int, required=True, help="Number of GPUs for tokensN")
    args = ap.parse_args()
    eff = (args.tokensN / (args.N * args.tokens1)) * 100.0
    print(f"Scaling efficiency: {eff:.2f}% (N={args.N}, 1-GPU={args.tokens1}, N-GPU={args.tokensN})")
if __name__ == "__main__":
    main()

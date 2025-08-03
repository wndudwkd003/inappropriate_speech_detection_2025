import os
import json
import matplotlib.pyplot as plt

class TrainResult:
    def __init__(self, train_output, log_history):
        # TrainOutput의 모든 속성 복사
        self.global_step = train_output.global_step
        self.training_loss = train_output.training_loss
        self.metrics = train_output.metrics
        # 로그 히스토리 추가
        self.log_history = log_history

def save_metrics(results, output_dir):
    """Save training results to a file and create loss plot."""

    # 기존 JSON 저장
    metrics_file = os.path.join(output_dir, "results.json")
    metrics_dict = {
        'final_train_loss': getattr(results, 'training_loss', None),
        'global_step': getattr(results, 'global_step', None),
        'train_runtime': getattr(results, 'results', {}).get('train_runtime', None),
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    # 로그 히스토리가 있으면 그래프 생성
    if hasattr(results, 'log_history') and results.log_history:

        # train loss와 eval loss 분리
        train_losses = []
        train_steps = []
        eval_losses = []
        eval_steps = []

        for log in results.log_history:
            if 'loss' in log and 'step' in log:
                train_losses.append(log['loss'])
                train_steps.append(log['step'])
            if 'eval_loss' in log and 'step' in log:
                eval_losses.append(log['eval_loss'])
                eval_steps.append(log['step'])

        # 그래프 생성
        if train_losses or eval_losses:
            plt.figure(figsize=(10, 6))

            if train_losses:
                plt.plot(train_steps, train_losses, 'b-', label='Train Loss', linewidth=2)

            if eval_losses:
                plt.plot(eval_steps, eval_losses, 'r-', label='Eval Loss', linewidth=2, marker='o')

            plt.title('Training and Evaluation Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 저장
            plot_file = os.path.join(output_dir, "loss_plot.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Loss plot saved to {plot_file}")
        else:
            print("No loss data found in log history")
    else:
        print("No log history found - skipping plot generation")

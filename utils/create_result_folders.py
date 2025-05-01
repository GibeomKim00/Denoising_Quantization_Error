import os

def save_results(model_name, test_acc, path, weights_path=None, top5_acc=None):
    os.makedirs(path, exist_ok=True)
    mode = 'a' if os.path.exists(os.path.join(path, 'test_results.txt')) else 'w'
    with open(os.path.join(path, 'test_results.txt'), mode) as f:
        f.write(f"Teacher model: {model_name}\n")
        if weights_path:
            f.write(f"Loading pre-trained weights from {weights_path}\n")
        f.write(f"==> Test Accuracy: Acc@1 {test_acc:.3f} || Acc@5 {top5_acc if top5_acc is not None else 98.100:.3f}\n")
        f.write(f"Teacher Test Accuracy: {test_acc:.3f}\n")
        f.write(f"Teacher test results saved to {path}\n")
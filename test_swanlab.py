import swanlab
import random

# 创建一个SwanLab项目
swanlab.init(
    project="my-awesome-project",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10
    }
)

epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    swanlab.log({"acc": acc, "loss": loss})

swanlab.finish()

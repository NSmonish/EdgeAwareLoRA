# EdgeAwareLoRA: Parameter-Efficient Fine-Tuning for Edge Devices

**EdgeAwareLoRA** is a custom PyTorch module that combines Low-Rank Adaptation (LoRA) with a Mixture-of-Experts (MoE) architecture to heavily compress Large Language Models for edge device deployment.

## 🚀 The Challenge & Solution
Deploying models like RoBERTa-Large on edge devices is highly constrained by memory and compute. Standard fine-tuning updates hundreds of millions of parameters. 

This framework replaces standard linear layers in transformer blocks with a custom **MoE LoRA layer**. It uses a router to select expert pathways, allowing the model to dynamically allocate parameter updates based on the input sequence, drastically reducing the active parameter count during inference while maintaining high accuracy.

## 📊 Key Results
By evaluating this architecture on GLUE benchmarks (MNLI, SST-2, MRPC, CoLA), this implementation achieved:
* **0.18% Trainable Parameters:** Reduced trainable parameters to just 638,976 out of 356M+.
* **High Accuracy:** Achieved **90.81% accuracy on MNLI** and **~96.2% on SST-2**.
* **Memory Efficiency:** Implemented memory-efficient Scaled Dot-Product Attention (SDPA) and a custom load-balancing (`LBTrainer`) loss function to prevent expert collapse.

## 🧠 Architecture Overview
The `EdgeAwareLoRALinear` module replaces a standard `nn.Linear` layer. It consists of:
1. **Router:** A linear layer that computes logits to select the top-k experts.
2. **Shared LoRA (A & B):** A base pathway that learns general representations.
3. **Expert LoRAs (A & B):** Specialized pathways that the router dynamically activates per sequence.
4. **Load Balancing Loss:** An auxiliary loss coefficient (`lb_coef`) added to the training loop to ensure all experts are utilized equally, preventing routing bottlenecks.

## 🛠️ Usage / Setup
The core logic and training loops are available in the Jupyter Notebook:
* [`EdgeAwareLoRA_MoE.ipynb`](./EdgeAwareLoRA_MoE.ipynb) - Contains the custom PyTorch classes, trainer modifications, and GLUE benchmark evaluation loops across multiple random seeds. 

*Built using PyTorch and HuggingFace Transformers.*

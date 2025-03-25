# E2-NVM: AI-Driven Energy-Efficient Non-Volatile Memory Management  

This repository contains code and experimental results related to **E2-NVM**, a software-level memory-aware storage layer designed to **improve energy efficiency and write endurance of Non-Volatile Memories (NVMs) using Variational Autoencoders (VAEs)**.  

### 🔬 **Research Background**  
This work is based on my **paper published at EDBT 2023** ([EDBT 2023 Conference Website](http://edbticdt2023.cs.uoi.gr/)):  
**[E2-NVM: A Memory-Aware Write Scheme to Improve Energy Efficiency and Write Endurance of NVMs](https://tuzijun111.github.io/paper/E2_NVM.pdf)**.  

The project introduces a deep learning-driven approach for **bit flip reduction**, leveraging VAEs and clustering techniques to **intelligently allocate memory segments** based on data similarity. Through real evaluations on **Intel Optane Memory**, E2-NVM demonstrated **up to 56% reduction in energy consumption**.  

### 🚀 **Ongoing Work: Incremental Privacy in E2-NVM**  
This repository also includes my **ongoing research** into **incremental privacy mechanisms** for E2-NVM. The goal is to extend the system with **privacy-aware memory optimizations**, ensuring **secure and efficient NVM management** in **modern edge-cloud environments**.  

---

## 📂 **Repository Structure**  

- `E2_NVM.ipynb` → Jupyter notebook with core implementations and results from the paper.  
- `models/` → Trained Variational Autoencoder (VAE) and clustering models.  
- `experiments/` → Scripts to reproduce key results, including bit flip reduction and energy efficiency benchmarks.  
- `incremental_privacy/` → Code for the new **privacy-aware extension** to E2-NVM (in progress).  

---

## 📊 **Key Contributions**  

✅ **AI-Driven Storage Optimization**: Uses VAEs and clustering to **reduce bit flips and extend NVM lifespan**.  
✅ **Real-World Evaluations**: Tested on **Intel Optane**, achieving **up to 56% energy savings**.  
✅ **Extensibility**: Can be integrated with **existing indexing solutions** and enhanced with **privacy-aware mechanisms**.  
✅ **Ongoing Research**: Incremental privacy mechanisms to secure **memory allocation in edge-cloud environments**.  

---

## 📜 **Citation**  

If you use this work, please cite the following paper:  

```bibtex
@inproceedings{kargar2023e2nvm,  
  author = {Saeed Kargar and Binbin Gu and Sangeetha Abdu Jyothi and Faisal Nawab},  
  title = {E2-NVM: A Memory-Aware Write Scheme to Improve Energy Efficiency and Write Endurance of NVMs},  
  booktitle = {Proceedings of the 26th International Conference on Extending Database Technology (EDBT)},  
  year = {2023}  
}
```

---

## 📩 **Contact**  

For any questions, discussions, or collaborations, feel free to reach out!  

👤 **Saeed Kargar**  
📧 [saeed.kargar@gmail.com](mailto:saeed.kargar@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/saeed-kargar-23498a1ba/)  

---

### ⭐ **If you find this project interesting, consider giving it a star!** ⭐  

---



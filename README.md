I recently built a Face Mask Detection model using transfer learning with ResNet, and the most valuable part wasn’t the final accuracy — it was understanding the pipeline.

🔹 Manually split raw image folders into train / validation / test to avoid data leakage
🔹 Used ImageFolder + DataLoader instead of train_test_split for scalable image loading
🔹 Froze pretrained layers and trained only the classifier head first
🔹 Evaluated using validation accuracy and confusion matrix, not just training accuracy

This project helped me understand how CNNs are actually used in real-world workflows, not just notebooks.

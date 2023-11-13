# 3DECG-Net
This repository encompasses the materials pertaining to the 3DECG-Net study. Within this study, a pioneering preprocessing framework denoted as 123ECG has been developed for the purpose of processing 12-lead electrocardiogram (ECG) recordings. Additionally, a novel 3D deep learning model has been devised to classify 12-lead ECG signals into seven distinct heart statuses, namely Normal Sinus Rhythm (NSR), Atrial Fibrillation (AF), First-Degree Atrioventricular Block (I-AVB), Left Bundle Branch Block (LBBB), Right Bundle Branch Block (RBBB), Sinus Bradycardia (SB), and Supraventricular Tachycardia (STach).

The 123ECG framework exhibits the capability to process 12-lead ECG recordings while ensuring the provision of high-level recordings without compromising the quality of the signals. This feature proves particularly advantageous in practical scenarios where storage hardware is constrained, and there exists an imperative demand for high-quality, lightweight data.

The 3DECG-Net model excels in its ability to classify heart statuses in a multi-label fashion, surpassing other state-of-the-art models in the domain. The demonstrated proficiency of 3DECG-Net positions it as a viable candidate for clinical applications, further solidifying its potential utility in the field.




`models.py` contains the architecture of 3DECG-Net and benchmark models

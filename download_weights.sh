#!/usr/bin/env bash
set -e

echo "========== Download PPLM weights =========="
if wget --spider -q "https://zhanggroup.org/PPLM/bin/weights/pplm_t33_650M.pt"; then
    echo "   File found on zhanggroup.org. Downloading with wget..."
    wget "https://zhanggroup.org/PPLM/bin/weights/pplm_t33_650M.pt"
else
    echo "   File not accessible on zhanggroup.org. Downloading from Google Drive..."
    gdown --fuzzy "https://drive.google.com/file/d/1teZBp3m_OQ4nciTmiepUDP8p9BXoVEwa/view?usp=drive_link"
fi

echo "========== Download PPLM-PPI weights =========="
if wget --spider -q "https://zhanggroup.org/PPLM/bin/weights/ppi_models.pkl"; then
    echo "   File found on zhanggroup.org. Downloading with wget..."
    wget "https://zhanggroup.org/PPLM/bin/weights/ppi_models.pkl"
else
    echo "   File not accessible on zhanggroup.org. Downloading from Google Drive..."
    gdown --fuzzy "https://drive.google.com/file/d/1Xdb3SG0CRY49WqH4jUJhM-yqsLOejz7_/view?usp=share_link"
fi

echo "========== Download PPLM-Affinity weights =========="
if wget --spider -q "https://zhanggroup.org/PPLM/bin/weights/affinity_models.pkl"; then
    echo "   File found on zhanggroup.org. Downloading with wget..."
    wget "https://zhanggroup.org/PPLM/bin/weights/affinity_models.pkl"
else
    echo "   File not accessible on zhanggroup.org. Downloading from Google Drive..."
    gdown --fuzzy "https://drive.google.com/file/d/1s99QyTYjngRUUpy8VXJPhTXYHwmw9agp/view?usp=share_link"
fi

echo "========== Download PPLM-Contact weights =========="
if wget --spider -q "https://zhanggroup.org/PPLM/bin/weights/pplm_contact_models.pkl"; then
    echo "   File found on zhanggroup.org. Downloading with wget..."
    wget "https://zhanggroup.org/PPLM/bin/weights/pplm_contact_models.pkl"
else
    echo "   File not accessible on zhanggroup.org. Downloading from Google Drive..."
    gdown --fuzzy "https://drive.google.com/file/d/1QxSFXojCQmLzgrTz398lUEdIZmqVVV9E/view?usp=share_link"
fi

echo "========== Download PPLM-Contact2 weights =========="
if wget --spider -q "https://zhanggroup.org/PPLM/bin/weights/pplm_contact_models2.pkl"; then
    echo "   File found on zhanggroup.org. Downloading with wget..."
    wget "https://zhanggroup.org/PPLM/bin/weights/pplm_contact_models2.pkl"
else
    echo "   File not accessible on zhanggroup.org. Downloading from Google Drive..."
    gdown --fuzzy "https://drive.google.com/file/d/1SSEkfyiwtUVO4ZSN10HC5T2v-EGOHXle/view?usp=share_link"
fi

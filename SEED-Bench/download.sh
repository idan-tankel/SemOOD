#!/bin/bash
fileid="1jgSoof1AatiDRpGY091qd4TEKF-BUt6I"
filename="breakfast_dataset_procedure_understanding.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
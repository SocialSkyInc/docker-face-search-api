echo 'only .jpg or .png (in lower case) are learn'
echo 'need multiple face (directory)'


cd /root/openface

rm -r /home/face_search_api/aligned
rm -r /home/face_search_api/brain
mkdir -p /home/face_search_api/aligned
mkdir -p /home/face_search_api/brain


echo '' && echo '' && echo ''
echo 'Preprocess the raw images'
./util/align-dlib.py /home/face_search_api/source-training align outerEyesAndNose /home/face_search_api/aligned --size 96
./util/prune-dataset.py /home/face_search_api/aligned --numImagesThreshold 1


echo '' && echo '' && echo ''
echo 'Generate Representations'
./batch-represent/main.lua -outDir /home/face_search_api/brain -data /home/face_search_api/aligned


echo '' && echo '' && echo ''
echo 'Create the Classification Model'
./demos/classifier.py train /home/face_search_api/brain



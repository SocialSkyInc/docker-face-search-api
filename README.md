# API face search

Based on Openface project.


## Setup example

```
docker run -d \
    -v /home/$USER/docker-data/face-search-api/source-training:/home/face_search_api/source-training \
    --name face-search-api \
    treemo/face-search-api
```


## Call example

```
curl -X POST http://127.0.0.1:8000/find -F "img=@/home/$USER/docker-data/face-search-api/source-training/einstein/1.png"
```

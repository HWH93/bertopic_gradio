import gradio as gr
import pandas as pd
import re
import numpy as np
import os
from datetime import datetime
from bertopic import BERTopic
import umap
import hdbscan
from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def topic(csv_file):

    global contents_data
    global timestamps
    
    # Data load
    lucy = pd.read_csv(csv_file.name, delimiter=",")
    lucy.fillna(0, inplace=True)
    lucy['contents'] = lucy['제목'].astype(str)+lucy['내용'].astype(str)
    lucy['contents'] = lucy['contents'].replace(np.nan, '')
    lucy['contents'] = lucy['contents'].replace("\n", '')
    lucy.drop_duplicates(['contents'], ignore_index=True)
    lucy['contents'] = [str(line).strip() for line in lucy['contents']]
    lucy['timestamp'] = [datetime.strptime(str(int(line)), '%Y%m%d') for line in lucy['수집일']]

    # Data filter
    lucy.contents = lucy.apply(lambda row: " ".join(re.sub("[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]", " ", row.contents).split()), 1)
    lucy.contents = lucy.apply(lambda row: " ".join(re.sub("([ㄱ-ㅎㅏ-ㅣ]+)", " ", row.contents).split()), 1)
    lucy.contents = lucy.apply(lambda row: " ".join(re.sub("([♡❤✌❣♥ᆢ✊❤️✨⤵️☺️;”“]+)", " ", row.contents).split()), 1)
    lucy.contents = lucy.apply(lambda row: " ".join(re.sub("_x000D_", "", row.contents).split()), 1)
    timestamps = lucy.timestamp.to_list()
    contents_data = lucy.contents.to_list()

    row_5 = lucy.iloc[[0,1,2,3,4],:]

    return row_5

def model(n_neighbors, min_cluster_size, min_samples):
    ## 모델 load 및 fine-tuning

    global topic_model

    n_neighbors = int(n_neighbors)
    min_cluster_size = int(min_cluster_size)
    min_samples = int(min_samples)

    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    corpus_embeddings = embedding_model.encode(contents_data, convert_to_tensor=True)
    corpus_embedding = corpus_embeddings.cpu()

    umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=0.0, metric='cosine')
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    topic_model = BERTopic(verbose=True, 
                     language="korean", 
                     umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, 
                     calculate_probabilities=True).fit(contents_data, corpus_embedding.numpy())

    message="Topic Model's Tuning is Done!"
    
    return message

def vt_topic():

    visual_topic = topic_model.visualize_topics()

    return visual_topic

def bc_topic(top_topic_num_bc):

    top_topic_num_bc = int(top_topic_num_bc)
    bar_chart = topic_model.visualize_barchart(top_n_topics=top_topic_num_bc, height=300)

    return bar_chart

def hi_topic(top_topic_num_hi):

    top_topic_num_hi = int(top_topic_num_hi)
    hierarchy = topic_model.visualize_hierarchy(top_n_topics=top_topic_num_hi, height=300)

    return hierarchy

def hm_topic(n_cluster):

    n_cluster = int(n_cluster)
    heatmap = topic_model.visualize_heatmap(n_clusters=n_cluster, top_n_topics=15)

    return heatmap

def tot_topic():

    topics_over_time = topic_model.topics_over_time(docs=contents_data, 
                                                timestamps=timestamps, 
                                                global_tuning=True, 
                                                evolution_tuning=True, 
                                                nr_bins=20)

    tot = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=15)

    return tot

def make_blank():

    gr.Markdown("#   ")
    gr.Markdown("#   ")

    return 

with gr.Blocks() as bert:
    gr.Markdown("# Lucy Topic Modeling")

    make_blank()

    csv_file = gr.File(label="입력 데이터 [*.csv]")  

    make_blank()

    with gr.Row():
        examples = gr.Examples(examples=["./data/토픽모델링_테스트_데이터_추출_멜론.csv"],inputs=[csv_file],fn=topic)
        exam = gr.Button("예제 시작")

    make_blank()

    gr.Markdown("### 입력 데이터 요약")
    row_5 = gr.Dataframe(show_label=False,type="pandas")

    make_blank()

    with gr.Row():
        with gr.Column(scale=1):
            n_neighbors = gr.Number(value=15,label="이웃할 포인트의 개수 입력")
        with gr.Column(scale=2):
            min_cluster_size = gr.Number(value=10,label="최소 클러스터의 크기 입력")
        with gr.Column(scale=3):
            min_samples = gr.Number(value=10,label="이웃할 최소 표본 개수 입력[≤최소 클러스터의 크기]")
        with gr.Column(scale=4):
            b1 = gr.Button("토픽 모델링 시작")
    
    exam.click(topic, csv_file, row_5)

    make_blank()

    csv_file.change(topic, csv_file, row_5)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 토픽 모델 튜닝 완료시 ")
            gr.Markdown("## 오른쪽 박스에 메세지 출력")
        with gr.Column(scale=2):
            message = gr.Textbox(show_label=False)

    b1.click(model,inputs=[n_neighbors,min_cluster_size,min_samples], outputs=[message])

    make_blank()
    
    gr.Markdown("### ================================================== 아래는 모델 튜닝 메시지 출력 이후 진행바랍니다. ==================================================")

    make_blank()

    with gr.Tab("Intertopic Distance Map"):
        vt = gr.Button("Intertopic Distance Map")
        visual_topic = gr.Plot()
    with gr.Tab("Topic Word Scores"):
        with gr.Row():
            top_topic_num_bc = gr.Number(value=5,label="Topic Word Scores로 출력할 주제 개수 입력")
            bc = gr.Button("Topic Word Scores")
        bar_chart = gr.Plot()
    with gr.Tab("Hierarchical Clustering"):
        with gr.Row():
            top_topic_num_hi = gr.Number(value=15,label="Hierarchical Clustering로 출력할 주제 개수 입력")
            hi = gr.Button("Hierarchical Clustering")
        hierarchy = gr.Plot()
    with gr.Tab("Similarity Matrix"):
        with gr.Row():
            n_cluster = gr.Number(value=5,label="Similarity Matrix의 최대 클러스터 개수 입력(주제 개수 보다 적게)")
            hm = gr.Button("Similarity Matrix")
        heatmap = gr.Plot()
    with gr.Tab("Topics Over Time"):
        tot = gr.Button("Topics Over Time")
        time = gr.Plot()

    vt.click(vt_topic,outputs=[visual_topic])
    bc.click(bc_topic,inputs=[top_topic_num_bc],outputs=[bar_chart])
    hi.click(hi_topic,inputs=[top_topic_num_hi],outputs=[hierarchy])
    hm.click(hm_topic,inputs=[n_cluster],outputs=[heatmap])
    tot.click(tot_topic,outputs=[time])

if __name__ == "__main__":
    bert.launch(enable_queue=False, server_name="0.0.0.0",server_port=8888)
# Sentiment-analysis-Vietnamese

wonrax/phobert-base-vietnamese-sentiment
một mô hình học sâu sử dụng PhoBERT (phiên bản tiếng Việt của RoBERTa), 
được fine-tune cho bài toán phân loại cảm xúc (sentiment analysis)


**Dataset:** [30K e-commerce reviews](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst)
gồm các câu được đánh nhãn với 3 label. 


**Cách train:**
đầu vào : câu text => qua model => output: xác suất dự đoán 3 nhãn
tính loss với nhãn đúng trong dataset bằng CrossEntropyLoss()
=> tối ưu model , cập nhật lại tham số model


**Kiến trúc mô hình:** (cơ bản vẫn là kiến trúc transformers)

RobertaForSequenceClassification(
  (roberta): RobertaModel(

    (embeddings): RobertaEmbeddings(
      (word_embeddings): Embedding(64001, 768, padding_idx=1)
      (position_embeddings): Embedding(258, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )

    (encoder): RobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x RobertaLayer(
          (attention): RobertaAttention(
            (self): RobertaSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): RobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): RobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): RobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (classifier): RobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=3, bias=True)
  )
  
**Phân tích kiến trúc:**
1. mô hình nền tảng (RobertaModel):

- Embeddings Layer
 + word_embeddings: 
Mảng embedding cho từ vựng, có kích thước (64001, 768). 
Mỗi từ trong từ vựng được biểu diễn bằng một vector 768 chiều.
 + position_embeddings: 
Các embedding cho vị trí của từ trong câu, giúp mô hình hiểu được thứ tự của các từ. Kích thước là (258, 768) (258 là số lượng vị trí có thể có trong câu).
 + token_type_embeddings: 
Dành cho loại token (sử dụng trong các tác vụ như câu hỏi trả lời, không cần thiết cho sentiment analysis nhưng vẫn có trong mô hình).
 + LayerNorm: 
Normal hóa layer giúp ổn định quá trình học.
 + dropout: 
Dropout với tỷ lệ 0.1 để tránh overfitting trong quá trình huấn luyện

- Encoder
RobertaEncoder chứa một chuỗi các lớp RobertaLayer (12 lớp trong trường hợp này), là các lớp mã hóa chính của mô hình. Mỗi lớp này bao gồm:
 + Self-Attention Mechanism: Chú ý tới các phần của câu (bằng cách sử dụng các trọng số cho query, key, và value) để tạo ra các biểu diễn ngữ nghĩa cho từng từ.
 + RobertaSelfOutput: Đầu ra của lớp self-attention, bao gồm một lớp dense để biến đổi các biểu diễn này, và layer normalization cùng với dropout.
 + Intermediate Layer: Một lớp hidden với kích thước 3072, sử dụng hàm kích hoạt GELU (Gaussian Error Linear Unit).
 + Output Layer: Đưa ra biểu diễn cuối cùng cho các từ sau khi qua các lớp attention và dense.

2. được train với dữ liệu tiếng Việt => model: PhoBERT 

3. fine-tune ở lớp Classifier (Phân loại): => model: phobert-base-vietnamese-sentiment

Classifier (Phân loại):

- Dense Layer: 
Đây là lớp fully connected (linear layer) với kích thước đầu vào là 768 và đầu ra là 768, 
giúp mô hình học các đặc trưng tổng hợp từ các lớp encoding trước đó.

- Dropout: 
dropout được áp dụng để giảm overfitting.

- Out Projection Layer: 
Lớp cuối cùng chuyển đổi từ không gian 768 chiều xuống không gian số lớp mục tiêu 
(ở đây là 3, tương ứng với ba nhãn cảm xúc). 
Từ đó, mô hình sẽ đưa ra dự đoán cho một trong ba lớp: tích cực, tiêu cực hoặc trung lập.

**Cách đánh giá %:**
với đầu ra là mảng 3 chiều, cho qua hàm softmax thì sẽ cho xác suất/ tỉ lệ tương ứng với các label
=> lấy ra giá trị và index của label có xác suất cao nhất

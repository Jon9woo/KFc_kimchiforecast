import pymysql

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from sklearn.preprocessing import MinMaxScaler
import os
from dotenv import load_dotenv
from tqdm import tqdm

# .env 파일에서 환경 변수 로드
load_dotenv()

# .env 파일에서 값을 가져옴
host = os.getenv('DB_HOST')
port = int(os.getenv('DB_PORT'))
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
database = os.getenv('DB_NAME')



####################################################################################################
###### 데이터베이스 연결 및 데이터 가져오기 ######
####################################################################################################

def connect_to_db(host, port, username, password, database):
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("Successfully connected to the database!")
        return connection

    except pymysql.MySQLError as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_data(connection, item):
    try:
        with connection.cursor() as cursor:
            item = item
            # cabbage, dried_pepper, garlic, ginger, green_onions, red_pepper
            sql_query = f"SELECT * FROM {item};"
            cursor.execute(sql_query)
            result = cursor.fetchall()
            df = pd.DataFrame(result)
            return df
    except:
        print("Error fetching data from the database!")
        return None
    #finally:
    #    if connection:
    #        connection.close()

####################################################################################################
###### 데이터 전처리 및 모델 학습 ######
####################################################################################################

def preprocess_data(df):
    df[['year', 'month', '순']] = df['date'].str.split('-', expand=True)
    
    순_order = {'Early': 1, 'Mid': 2, 'Late': 3}
    df['순_numeric'] = df['순'].map(순_order)

    month_order = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    df['month_numeric'] = df['month'].map(month_order)
    df['year'] = df['year'].astype(int)
    df_sorted = df.sort_values(by=['year', 'month_numeric', '순_numeric'])

    data = df_sorted[['date', 'data']].reset_index(drop=True)
    
    return data

def process_data(raw_data, scaler=None):
    filtered_data = raw_data[['date', 'data']]
    numeric_columns = ['data']
    filtered_data[numeric_columns] = filtered_data[numeric_columns].fillna(0)

    if scaler is None:
        scaler = MinMaxScaler()
        filtered_data[numeric_columns] = scaler.fit_transform(filtered_data[numeric_columns])
    else:
        filtered_data[numeric_columns] = scaler.transform(filtered_data[numeric_columns])

    return filtered_data, scaler

class AgriculturePriceDataset(Dataset):
    def __init__(self, dataframe, window_size=27, prediction_length=9, is_test=False):
        self.data = dataframe
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.is_test = is_test
        
        self.price_column = 'data'
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.sequences = []
        if not self.is_test:
            for i in range(len(self.data) - self.window_size - self.prediction_length + 1):
                x = self.data[self.numeric_columns].iloc[i:i+self.window_size].values
                y = self.data[self.price_column].iloc[i+self.window_size:i+self.window_size+self.prediction_length].values
                self.sequences.append((x, y))
        else:
            self.sequences = [self.data[self.numeric_columns].values]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if not self.is_test:
            x, y = self.sequences[idx]
            return torch.FloatTensor(x), torch.FloatTensor(y)
        else:
            return torch.FloatTensor(self.sequences[idx])

class PricePredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PricePredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def run_training(dataset, val_data, CFG, item):
    # numeric_columns은 dataset의 초기화 시 이미 설정됨
    input_size = len(dataset[0][0][0])  # 첫 번째 샘플의 입력 크기
    
    model = PricePredictionLSTM(input_size, CFG.hidden_size, CFG.num_layers, CFG.output_size)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)

    train_loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CFG.batch_size, shuffle=False)

    best_val_loss = float('inf')
    os.makedirs('../models', exist_ok=True)

    for epoch in range(CFG.epoch):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss = evaluate_model(model, val_loader, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'../models/best_model_{item}.pth')
        
        print(f'Epoch {epoch+1}/{CFG.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return model, best_val_loss

####################################################################################################
###### 데이터 예측 및 데이터베이스 저장 ######
####################################################################################################

if __name__ == "__main__":
    config = {
        "learning_rate": 2e-4,
        "epoch": 100,
        "batch_size": 64,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 9    
    }
    CFG = SimpleNamespace(**config)

    # DB 연결
    connection = connect_to_db('192.168.101.37', 3306, 'futures', '1234', 'price_predict_db')
    
    # 데이터 가져오기
    list = ['cabbage', 'dried_pepper', 'garlic', 'ginger', 'green_onions', 'red_pepper']
    list_val = {}
    for item in list:
        print('-'*50)
        print(f"Processing {item} data...")
        print('-'*50)
        if connection:
            df = fetch_data(connection, item=item)
            data = preprocess_data(df)

            # 마지막 27개는 테스트 데이터로 사용
            train_data = data[:-36]
            test_data = data[-36:-9]

            # 데이터 전처리
            train_data, scaler = process_data(train_data)
            dataset = AgriculturePriceDataset(train_data)

            # 훈련 및 검증 데이터 분할
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

            # 모델 학습
            model, best_val_loss = run_training(train_data, val_data, CFG, item=item)
            print(f"Best Validation Loss: {best_val_loss:.4f}")
            list_val[item] = best_val_loss

            # 테스트 데이터 예측
            test_data, _ = process_data(test_data, scaler=scaler)
            test_dataset = AgriculturePriceDataset(test_data, is_test=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    output = model(batch)
                    predictions.append(output.numpy())

            predictions_array = np.concatenate(predictions)
            predictions_reshaped = predictions_array.reshape(-1, 1)
            predictions_original_scale = scaler.inverse_transform(predictions_reshaped)

            print(predictions_original_scale)
        
        # 예측값을 데이터베이스에 저장
        try:
            with connection.cursor() as cursor:
                for i, prediction in enumerate(predictions_original_scale):
                    # 테이블 생성 쿼리
                    sql_query = f"""
                    CREATE TABLE IF NOT EXISTS {item}_prediction (
                    date VARCHAR(255),
                    price FLOAT
                    );
                    """
                    cursor.execute(sql_query)  # 쿼리 실행

                    # 데이터 삽입 쿼리
                    date = test_data['date'].iloc[i]
                    print(date)
                    price = prediction[0]
                    print((price))
                    sql_query = f"INSERT INTO {item}_prediction VALUES ('{date}', {price});"
                    cursor.execute(sql_query)
                connection.commit()
        except:
            print("Error inserting data to the database!")
    
    if connection:
        connection.close()

    predictions_original_scale
    print(list_val)
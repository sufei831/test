import numpy as np


# 假设函数来计算概率
def get_Pdh(data, feature, class_value):
    # 计数特征值在每个类别下的出现次数  
    counts={}
    count=0
    for example in data:
        if example['PlayTennis'] == class_value:
            count+=1
            feature_value=example[feature]
            if feature_value not in counts:
                counts[feature_value]=0
            counts[feature_value]+=1

            # 添加平滑
    smooth=1
    count+=len(counts) * smooth
    for key in counts:
        counts[key]+=smooth

        # 计算概率
    p = {k: v / count for k, v in counts.items()}
    return p


# 计算先验概率
def get_Ph(data):
    c_yes=sum(1 for example in data if example['PlayTennis'])
    c_no=len(data)-c_yes
    total_count=len(data)
    r_y=(c_yes+1)/(total_count+2)
    r_n=(c_no+1)/(total_count+2)
    return r_y ,r_n


# 朴素贝叶斯分类器函数
def Bayes(train_data, test_data):
    # 计算先验概率  
    p_y,p_n=get_Ph(train_data)

    # 计算每个特征在每个类别下的条件概率  
    features={}
    for feature in ['Outlook', 'Temperature', 'Humidity', 'Wind']:
        features[feature] = {
            'positive': get_Pdh(train_data, feature, True),
            'negative': get_Pdh(train_data, feature, False)
        }

        # 预测
    predictions=[]
    for example in test_data:
        V_y=p_y
        V_n=p_n
        for feature in ['Outlook', 'Temperature', 'Humidity', 'Wind']:
            feature_value = example[feature]
            p_yf = features[feature]['positive'].get(feature_value, 0)
            p_nf = features[feature]['negative'].get(feature_value, 0)
            V_y *= p_yf
            V_n *= p_nf

            # 选择概率更大的类别
        if V_y>V_n:
            predictions.append(True)
        else:
            predictions.append(False)

    return predictions


if __name__=="__main__":
    examples = [
        # 将每一个训练样例组织成一个dict，包括目标属性（PlayTennis），便于在构建决策树过程中查询特定属性的值
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": False},
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "PlayTennis": False},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": True},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": True},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": True},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": False},
        {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": True},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": False},
        {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": True},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": True},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": True},
        {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": True},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": True},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": False}
    ]
    idx = list(range(len(examples)))
    np.random.shuffle(idx)
    train_data = [examples[i] for i in idx[:10]]
    #print(train_data)
    test_data = [examples[i] for i in idx[10:]]
    print(test_data)

    predictions = Bayes(train_data, test_data)
    print(predictions)

    true_labels = [example['PlayTennis'] for example in test_data]
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    print(f"Accuracy: {accuracy}")
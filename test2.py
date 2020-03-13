

if __name__ == "__main__":
    dict1 = {
        "123": {1976: 0.3, 1977: 0.4, 1978: 0.5}, 
        "246": {1935: 0.2}
    }

    save_list = []
    for k1, v1 in dict1.items():
        for k2, v2 in v1.items():
            temp_list = []
            temp_list.append(k1)
            temp_list.append(k2)
            temp_list.append(v2)
            save_list.append(temp_list)
    
    print(save_list)
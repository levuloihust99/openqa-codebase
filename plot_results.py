from matplotlib import pyplot as plt
import re


def read_data(file_path):
    data = []
    reader = open(file_path, "r")
    lines = reader.readlines()
    for line in lines:
        score = eval(line.split(':')[1].strip())
        data.append(score)
    
    reader.close()
    return data


def main():
    f1 = "results/single/top_k_hits.txt"
    f2 = "results/retriever/hardnegvsnegsoftmax_batch16_query32_gradnorm3/top_k_hits.txt"

    data_1 = read_data(f1)
    data_2 = read_data(f2)

    fig = plt.figure(figsize=(12,5))
    chart = fig.add_subplot()
    chart.set_xlabel("Top K")
    chart.set_ylabel("Hits")
    chart.plot(range(1, 101), data_1, label='baseline')
    chart.plot(range(1, 101), data_2, label='best tuned model')
    chart.legend()

    fig.savefig("compare.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
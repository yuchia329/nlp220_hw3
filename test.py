import json
from loaddata import loadJSON
df = loadJSON('arxiv_data.json')
# summaries = df['summaries']
import matplotlib.pyplot as plt
# df = loadJSON('token_freq_pie.json')
def plot_label_distribution(df, title):
    # Flatten the lists of labels (assumes 'terms' column contains lists)
    all_labels = df['terms'].explode()  # Expanding lists into individual elements
    label_counts = all_labels.value_counts()  # Count occurrences of each label
    total = sum(label_counts)
    label_counts = label_counts*100/total
    label_counts = label_counts[:10]
    # Print the labels and their counts to the terminal
    # print(f"\nLabel counts for in {title}:")
    # for label, count in label_counts.items():
    #     print(f"{label}: {count}")
    # print(f"Length of {title}: {len(df)}")

    # Plotting
    plt.figure(figsize=(12, 6))
    label_counts.plot(kind='bar')
    plt.title("Label Distribution in " + title)
    plt.xlabel('Labels')
    plt.ylabel('Frequency(%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.savefig('label_dist.jpg')

plot_label_distribution(df, 'Label Counts')
# x,y=splitXY(df)
# print(y.columns)
# label_freq = {}
# for col in y.columns:
#     label_freq[col] = sum(y[col])
# # keys = label_freq.keys()
# # labels = label_freq.values()


# def rank_json_key_by_value(json_data):

#     ranked_keys = sorted(json_data, key=json_data.get, reverse=True)
#     return ranked_keys

# ranked_keys = rank_json_key_by_value(label_freq)
# new_label_freq = {}
# for key in ranked_keys:
#     new_label_freq[key] = label_freq[key]
# print(ranked_keys)

# with open('token_freq.json', 'r') as f:
#     data = json.load(f)
#     total=sum(data.values())
#     new_data = {}
#     for key in data.keys():
#         new_data[key] = round(100*data[key]/total,3)
#     with open('token_freq_pie.json', 'w') as f:
#         json.dump(new_data, f)

# with open('token_freq.json', 'r') as f:
#     data = json.load(f)
#     total=sum(data.values())
#     new_data = {}
#     acc = 0
#     for key in data.keys():
#         acc += 100*data[key]/total
#         new_data[key] = round(acc,5)
#     with open('token_freq_acc.json', 'w') as f:
#         json.dump(new_data, f)
from openpyxl import load_workbook


label_vocab = ['contradiction', 'entailment', 'neutral']
tsv_file = open("../diin/eval_report.tsv", 'r')
with open("submission.csv", 'w', encoding='utf-8') as csv_file:
    csv_file.write("pairID,gold_label\n")
    rows = tsv_file.readlines()
    for row in rows[1:]:
        field = row.strip().split("\t")
        csv_file.write("%s,%s\n"%(field[0], label_vocab[int(field[1])]))


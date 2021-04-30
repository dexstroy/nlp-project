from openpyxl import load_workbook


# Opens a single sheet for reading, returns a list of messages and a list of classes
def read_worksheet(filename, sheet_name, all_classes, group_mapper, label_encoder, no_columns):
    wb = load_workbook(filename, read_only=True)
    ws = wb[sheet_name]
    column_labels = next(ws.rows)
    X = []
    y = []
    for row in ws.rows:
        if row[0].value is None:
            break
        elif row[0].value.strip() == "Course":  # Skip the first line which only contains column titles
            continue
        new_entry = {}
        for i in range(no_columns):
            new_entry[column_labels[i].value.lower().replace(" ", "_")] = str(row[i].value)
        c_list = [new_entry["codepreliminary"].lower().strip()]
        if c_list[0] not in all_classes and c_list[0] not in group_mapper.keys():
            c_list = new_entry["codepreliminary"].lower().strip().split("/")
        # if still not belongs to any of classes then belongs to some group class
        if c_list[0] not in all_classes:
            c_list = [group_mapper[c_list[0]]]
        # If there are 2 classes listed in document add message twice (1 for each class)
        for c in c_list:
            new_entry["codepreliminary"] = label_encoder.transform([c])[0]
            X.append(new_entry["message"])
            y.append(label_encoder.transform([c])[0])
    wb.close()
    return X, y
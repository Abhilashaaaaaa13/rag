import pandas as pd
import json

EXCEL_PATH = "DH Grouping updated final(1).xlsx"


# Helper function

def clean_text(val):
    if pd.isna(val):
        return ""
    return str(val).strip()



# GROUPS CHUNKS

def create_group_chunks(df):
    chunks = []

    for _, row in df.iterrows():
        group_id = clean_text(row["Group ID"])
        group_name = clean_text(row["Group Name"])
        description = clean_text(row["Description"])
        level = clean_text(row["Level"])

        alt_names = []
        for col in ["AlternateName1", "AlternateName2", "AlternateName3", "AlternateName4"]:
            if col in df.columns and clean_text(row[col]):
                alt_names.append(clean_text(row[col]))

        content = (
            f"Group Name: {group_name}. "
            f"Description: {description}. "
            f"Alternative Names: {', '.join(alt_names)}."
        )

        chunk = {
            "id": f"group_{group_id}",
            "type": "group",
            "content": content,
            "metadata": {
                "group_id": group_id,
                "group_name": group_name.lower(),
                "level": level
            }
        }

        chunks.append(chunk)

    return chunks



# USERS CHUNKS

def create_user_chunks(df):
    chunks = []

    for _, row in df.iterrows():
        username = clean_text(row["User name"])
        designation = clean_text(row["Designation"])
        hierarchy = clean_text(row["Hierarchy"])
        level = clean_text(row["Level"])
        appsavy_id = clean_text(row["Appsavy ID"])

        content = (
            f"User Name: {username}. "
            f"Designation: {designation}. "
            f"Hierarchy: {hierarchy}. "
            f"Level: {level}."
        )

        chunk = {
            "id": f"user_{appsavy_id}",
            "type": "user",
            "content": content,
            "metadata": {
                "user_name": username.lower(),
                "designation": designation,
                "hierarchy": hierarchy,
                "level": level,
                "appsavy_id": appsavy_id
            }
        }

        chunks.append(chunk)

    return chunks



# MAIN

def main():
    groups_df = pd.read_excel(EXCEL_PATH, sheet_name="Groups")
    users_df = pd.read_excel(EXCEL_PATH, sheet_name="Users")

    group_chunks = create_group_chunks(groups_df)
    user_chunks = create_user_chunks(users_df)

    all_chunks = group_chunks + user_chunks

    with open("rag_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks created: {len(all_chunks)}")
    print("Saved to rag_chunks.json")


if __name__ == "__main__":
    main()

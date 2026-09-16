from langchain_text_splitters import RecursiveCharacterTextSplitter
with open('./data/docs/employee_handbook.md', 'r') as file:
    content = file.read()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25)
chunks = splitter.split_text(content)     # 输入一个字符串，输出一个字符串列表

for i in chunks:
    print(len(i))
print(len(chunks))
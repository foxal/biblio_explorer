# Read Me
This file explains the Workflow of Social Network Analysis

## Data collection, pre-processing, and storage
- Retrieve bibliographic items from the following sources and save to xml files:
  - Create a script to retrieve bibliographic items from the NDL Search API [Link](https://ndlsearch.ndl.go.jp/help/api/specifications)
    - Get items in three modes: query by keywords in title, query by creator, or query by NDC code, the user can choose the mode and specify the time range by setting parameters: 
      - By keywords: https://ndlsearch.ndl.go.jp/api/sru?operation=searchRetrieve&version=1.2&recordPacking=xml&recordSchema=dcndl&query=title%3d%22%E3%82%B7%E3%82%B9%E3%83%86%E3%83%A0%22%20AND%20ndc%3d%22007%22%20AND%20from%3d%221950%22%20AND%20until%3d%221990%22
      - By creator: https://ndlsearch.ndl.go.jp/api/sru?operation=searchRetrieve&version=1.2&recordPacking=xml&recordSchema=dcndl&query=creator%3d%22%E6%9E%97%E9%9B%84%E4%BA%8C%E9%83%8E%22%20AND%20from%3d%221950%22%20AND%20until%3d%221990%22
      - By NDC code: https://ndlsearch.ndl.go.jp/api/sru?operation=searchRetrieve&version=1.2&recordPacking=xml&recordSchema=dcndl&query=ndc%3d%225%22%20AND%20from%3d%221950%22%20AND%20until%3d%221990%22
- Process the retrieved items and save to SQLite database
  - Extract data fields from xml files: dcndl:BibAdminResource (get ID from the URL in rdf:about, e.g. "https://ndlsearch.ndl.go.jp/books/R000000004-I842968"), itemType from dcterms:description of dcndl:BibAdminResource, title (dcndl:BibResource - dcterms:title), authors (dcterms:creator), publisher (dcterms:publisher), publication date (dcterms:date), subject (dc:subject), publication title if the type is "article" (dcndl:publicationName).
  - Remove duplications: Compare the item being processed with existing items in the database. If this item and an existing item has similar titles (or one's contains the other's), same authors (or one's contains the other's), and publication dates, then combine these two items. Data fields with more information (e.g. long title, more authors, earlier publication date, etc.) should be kept. When doing the comparison, semantic similarity is first used to compare the titles and authors. If the similarity is greater than 0.8, GPT-4o is then used to make more accurate judgments. If GPT feels it is difficult to make a decision, or the manual mode of comparison is on, the user should be asked to make a final decision.
  - Save the extracted data of unique items to SQLite database. The primary key should be the ID of the item.
- Create separate table for the entities (authors and publishers) and their metadata
  - Extract the names of authors and publishers from the extracted data and save to a separate table. The primary key should be an id based on the name of the entity. This id should then be used as a foreign key for referring to these entities in the bibliographic items' table.
  - Identify entities and their types (read data from a saved place (a csv file, or a SQLite table), if matched, use the matched type, otherwise, save new records)
    - if using csv, refer to utility_csv.py, which produces the "name_romaji_cleaned.csv" file
    - Store entities and their types in Zotero or SQLite database
      - If using SQLite, store the entities and their types in a separate table. (Preferred now)
      - If using Zotero, store the entities and their types in a separate csv file.
      

## Transferring data from SQLite/Zotero to Neo4j database
- Export data from Zotero to json if using Zotero. If using SQLite, this step can be omitted.

## Social Network Analysis using Neo4j


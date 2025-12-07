from neo4j import GraphDatabase

URI = "neo4j+ssc://dc428781.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "EF66wON70MGc3cmObBLN0DMq0ASZ71k78C4MocTg9kk"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
driver.verify_connectivity()
print("✅ Connected OK!")
driver.close()

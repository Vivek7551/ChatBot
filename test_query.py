import rag_engine
import json

engine = rag_engine.RAGEngine()
# Ensure store loads from the existing chunks
engine.load()

ans, chunks = engine.ask('According to the 1983 and 1984 papers, how do folic acid and adenosine differ in their effect on the cAMP adaptation mechanism?')
print('\n--- ANSWER ---\n')
print(ans)

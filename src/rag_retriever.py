def retrieve_context(query, model, index, texts, k=3):
    q_embed = model.encode([query])
    D, I = index.search(q_embed, k)
    return [texts[i] for i in I[0]]

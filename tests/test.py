from llm_sdk import Small_LLM_Model


def test_small_llm_model():
    model = Small_LLM_Model(device="cpu")
    max_res_tokens = 20
    tokens_ids = model.encode("What is the capital of France?")[0].tolist()
    response_tokens_ids: list[int] = []
    print(f"Response France: {tokens_ids}")

    for _ in range(max_res_tokens):
        loggits = model.get_logits_from_input_ids(tokens_ids)
        new_token_id = loggits.index(max(loggits))
        tokens_ids.append(new_token_id)
        response_tokens_ids.append(new_token_id)
        #print(f"Loggits: {loggits}")
        print(f"Next token id: {new_token_id}")
        response = model.decode([new_token_id])
        print(f"Decoded response: {response}")

    print(f"Final response token ids: {response_tokens_ids}")
    print(f"Final response: {model.decode(response_tokens_ids)}")


    tokens_ids = model.encode("What is the capital of Rome?")[0].tolist()
    print(f"Response Rome: {tokens_ids}")
 

if __name__ == "__main__":
    test_small_llm_model()

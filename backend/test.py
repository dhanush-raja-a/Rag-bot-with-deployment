from rag_pipeline import generate_answer
 
response = generate_answer("i am bleeding in urine part", top_k=3)
print(response)
# print("Query:", response["query"])
# print("Answer:", response["answer"])
# print("Sources:")

'''(base) dhanushrajaa@Dhanushs-MacBook-Air untitled folder % python3 test.py 
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/utils/_pytree.py:185: FutureWarning: optree is installed but the version is too old to support PyTorch Dynamo in C++ pytree. C++ pytree support is disabled. Please consider upgrading optree using `python3 -m pip install --upgrade 'optree>=0.13.0'`.
  warnings.warn(
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Query: What are the symptoms of diabetes?
Answer: The symptoms of diabetes include:

1. Being very thirsty
2. Frequent urination
3. Feeling very hungry or tired
4. Losing weight without trying
5. Having sores that heal slowly
6. Dry, itchy skin
7. Loss of feeling or tingling in the feet
8. Blurry eyesight

Some people with diabetes may not experience any symptoms at all. If you're concerned about diabetes, it's best to consult with a doctor who can perform a blood test to determine if you have the condition.
Sources:'''
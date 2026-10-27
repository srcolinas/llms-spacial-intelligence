# LLMs Spacial Intelligence

In this repo I create code to explore how LLMs deal with a question to test their ability to reason spacially. I create a hexagonal grid of size three where the vertices will represent positions and the edges will represent the path between two vertices. The task to the LLM is to figure the shortest path from a particular vertex to another one.

The task could be given to the LLM as a description of the coordinate system or as a graph problem, so the code is structured to test both scenarios. Specifically, given two vertices, one could ask:

* Tell me the shortest path from vertex (1,0,1) to vertex (2,1,0) (assuming a good description of the coordinate system).
* Tell me the shortest path from node E to node D.

Both questions are basically the same, but they convey context in different ways. Hopefully this can help to diagnose whether the conclusions are a consequence of the representation of the problem or whether they come from inherent spacial intelligence of the LLMs.

## Running the project

If you have already setup the project (by running `task setup`), run:

1. `task build` to compile the Rust workspace and install the Python extension.
2. `task create-dataset` to generate a dataset of question and answer pairs. It should generate `data/questions.jsonl` `data/graph.json` and `data/coordinates.md` files; the first is the questions, while the last two are descriptions of the space. 
3. `task view` to open the path visualizer in a browser.
4. `task solve -- --llm ollama:llama3.1:8b --agent bare --problem graph` to solve one problem with the llm and agent of choice. For the LLM, any string supported by Pydantic AI will work. Ensure credentials for the LLM are provided in the `.env` file.
5. `task evaluate -- --answers data/answers.graph.bare.ollama-llama3.1-8b.jsonl --answers data/answers.graph.bare.gateway-openai-gpt-6-luna.jsonl` to score one or more answer files and write `data/evaluation.jsonl`.
6. `task view` loads questions, answers, and evaluation from `data/` by default; pass `--questions`, `--answers`, or `--evaluation` to override individual files.

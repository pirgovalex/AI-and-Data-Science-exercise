1. YAML Parsing  (assignment_yaml_parsing.py)

The Task: Create a CLI tool to parse a YAML file and retrieve a value using a nested key (e.g., 'user.profile.name').

How I Did It: I used argparse to handle command-line inputs for the file path and the key. The real brain-teaser was writing the check_key_in_yaml function. It had to traverse the parsed data, handling both dictionaries and lists. I had to think carefully about the control flow: if the current level is a dict, check for the key. If it's a list, try to convert the key to an integer and use it as an index. I wrapped everything in try-except blocks to make it robust and give clear error messages.

What I Learned: So much about data structure traversal and the importance of validation. Making a script user-friendly isn't just about the core logic; it's about anticipating every way it could break and handling it gracefully. It really drilled into me the difference between a quick script and a reliable tool.

2. The Matrix Command Center (matrix_calculator.py)

The Task: Build a full-fledged CLI matrix calculator using NumPy.

How I Did It: I built an OOP-style Matrix class around a NumPy array. Each operation (add, multiply, inverse, solve, etc.) is a method. I even used a decorator @auto_print (which i find immensely satisfying) to automatically print out results during testing. The main() function uses argparse to let users specify an operation (like --operation dot) and paths to CSV files for input/output. The trickiest part was handling the different requirements for each op, like making sure --input2 is provided for binary operations.

What I Learned: The power of wrapping a powerful library like NumPy in a clean, intuitive interface. It reinforced my understanding of matrix algebra and how to structure a larger CLI application with clear, modular components. Seeing np.linalg.solve work its magic never gets old!

3. The Web Explorer & Its Controller (scraper.py & scraper_CLI.py)

The Task: Build a scraper to recursively crawl a website and extract clean text content.

How I Did It: The core of the scraper uses requests and BeautifulSoup. The key function, get_links, is recursive—it starts from a base URL, scrapes all links on the page that belong to the same domain, and then calls itself for each new link, keeping track of depth to avoid going too far. I implemented a simple blacklist to filter out navigational text like "Copyright" and "Privacy Policy." The separate CLI file lets us easily configure the starting URL and depth.

What I Learned: The practical challenges of web scraping: recursion limits, handling exceptions so the whole script doesn't crash on one bad link, and the eternal struggle of cleaning noisy HTML to get just the content you want. It was a great lesson in building a configurable tool.

4. The Neural Network Navigator (bc_predict.py)

The Task: Implement a binary classification neural network from scratch with NumPy.

How I Did It: This was huge! I built a simple 2-layer network (Input -> Hidden ReLU -> Output Sigmoid). The script implements forward propagation, backward propagation (calculating gradients like dW2 and db1), and parameter updating via gradient descent. It loads training/test data from CSVs, normalizes it, and trains over many epochs. The loss and accuracy slowly improve, which is incredibly satisfying to watch print out.

What I Learned: Everything. This project demystified neural networks for me. I now have a concrete understanding of what backpropagation actually does and how the gradients are used to nudge the weights and biases in the right direction. It’s one thing to know the math, it’s another to code the step dZ2 = A2 - Y and see it work.
What's Next? The Hybrid Search Retriever!

This has all been amazing prep for my final project of the internship: building a retriever using hybrid search for task 3. I'm now perfectly comfortable handling data (YAML, CSVs), building complex systems (OOP, CLIs), and understanding the core math behind AI models. I'm ready to combine dense vector embeddings with traditional keyword matching to build something really powerful. I'll be using everything I've learned there to make it efficient and user-friendly.

--TODO retriever
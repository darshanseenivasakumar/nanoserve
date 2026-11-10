"""Generate bench/prompts.jsonl, the fixed benchmark prompt set.

Run once; the output is committed so every engine and every milestone sees identical prompts:
    uv run python -m bench.make_prompts

Three length buckets so prefill cost is realistic:
  short  (~45-70 tokens after the chat template)   hand-written questions
  medium (~150-220 tokens)                          a task over one hand-written passage
  long   (~550-700 tokens)                          a task over FOUR passages joined into one document
SHORTCUT (flagged): long prompts are composites of the medium passages rather than independent texts.
That is fine for timing (prefill cost depends on length, not originality) but means long prompts share
content with medium ones. Prefix caching is disabled in every engine we compare, so this can't help anyone.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).with_name("prompts.jsonl")

SHORT = [
    "What is the capital of Australia, and why isn't it Sydney?",
    "Explain recursion to a ten-year-old in three sentences.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Give me five creative names for a bakery that only sells bread made with sourdough.",
    "What is the difference between a list and a tuple in Python?",
    "Translate 'Where is the nearest train station?' into French, Spanish and German.",
    "Why is the sky blue? Answer briefly.",
    "Write a haiku about debugging code late at night.",
    "List the planets of the solar system in order from the sun.",
    "What does HTTP status code 404 mean?",
    "Suggest a simple weekly workout plan for a beginner with no equipment.",
    "Explain what a hash table is and when you would use one.",
    "Write a SQL query that returns the ten most recent orders from an orders table.",
    "What are the main causes of inflation?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How do vaccines train the immune system?",
    "Write a short motivational message for a student before an exam.",
    "What is the time complexity of binary search, and why?",
    "Give three tips for writing clear technical documentation.",
    "Explain the difference between RAM and storage.",
    "Write a limerick about a cat who learns to code.",
    "How does a refrigerator keep food cold?",
    "What is gradient descent, in one paragraph?",
    "Convert 72 degrees Fahrenheit to Celsius and show the formula.",
]

# (passage, task) pairs. Passages are original text written for this benchmark.
PASSAGES = [
    ("The monsoon arrives in Kerala in early June, usually within a week of the first of the month. "
     "Farmers watch the sky and the forecasts closely because the timing decides when paddy is sown. "
     "A late monsoon can shorten the growing season, while an early and heavy one can flood low fields "
     "before seedlings are strong enough to survive. Reservoirs that fed cities through the dry months "
     "begin to refill, and power companies that depend on hydroelectric dams breathe easier. Meteorologists "
     "track sea surface temperatures in the Arabian Sea and pressure patterns over the Indian Ocean, but "
     "the exact onset still surprises them in some years, which is why the date is announced only after "
     "several days of steady rain across a network of stations."),
    ("A compiler translates a program written in a high-level language into instructions a machine can run. "
     "It usually works in stages. The lexer breaks source text into tokens such as keywords, numbers and "
     "names. The parser arranges those tokens into a tree that reflects the grammar of the language. Later "
     "stages check types, simplify the tree, and finally emit machine code or an intermediate form. Modern "
     "compilers spend much of their effort on optimization: removing work that has no effect, keeping values "
     "in registers instead of memory, and reordering instructions so the processor stays busy. Good error "
     "messages are also a major engineering challenge, because the compiler must guess what the programmer "
     "meant when the code is wrong."),
    ("Honeybees communicate the location of food through a movement called the waggle dance. A forager "
     "returning to the hive runs in a straight line while shaking her body, then loops back and repeats. "
     "The angle of the straight run relative to vertical tells the other bees the direction of the food "
     "relative to the sun, and the duration of the waggle encodes distance. Bees that follow the dance "
     "then fly out and search the indicated area, using smell to find the exact flowers. The dance adjusts "
     "for the movement of the sun during the day, which means bees carry an internal sense of time. "
     "Researchers have decoded these dances by filming hives and measuring the angles frame by frame."),
    ("Double-entry bookkeeping records every transaction in at least two accounts, as a debit in one and "
     "a credit in another, so that the totals always balance. If a shop buys stock with cash, its inventory "
     "account increases while its cash account decreases by the same amount. This simple rule makes many "
     "errors visible, because a mistake usually causes the books to stop balancing. The method spread "
     "through Italian merchant cities in the fifteenth century and was described in a widely read "
     "mathematics textbook of the time. Today accounting software applies the same principle automatically, "
     "but auditors still rely on it when they trace where money came from and where it went."),
    ("A graphics processing unit was originally designed to draw images, which requires applying the same "
     "small calculation to millions of pixels. To do this quickly, it contains thousands of simple cores "
     "that work in parallel, grouped so that many threads run the same instruction on different data. "
     "Researchers noticed that neural networks have the same shape of work: large matrix multiplications "
     "where every output element is computed the same way. GPUs therefore became the main hardware for "
     "training and running deep learning models. Their main limitation is memory: data has to be moved "
     "from memory into the cores, and for many workloads the speed of that movement, not the arithmetic, "
     "decides how fast the program runs."),
    ("Sourdough bread rises without commercial yeast. Instead, the baker keeps a starter, a mixture of flour "
     "and water colonised by wild yeasts and lactic acid bacteria. The yeasts produce carbon dioxide that "
     "inflates the dough, while the bacteria produce acids that give the bread its tangy flavour and help "
     "it keep longer. Maintaining a starter means feeding it fresh flour and water regularly and discarding "
     "part of it so the population stays active. Temperature matters a great deal: a warm kitchen speeds "
     "fermentation and increases sourness, while a cool one slows everything down. Many bakers let shaped "
     "loaves rest overnight in a refrigerator to develop flavour before baking them in a very hot oven."),
    ("Version control systems record the history of changes to a set of files. Each saved state, called a "
     "commit, stores who made the change, when, and a message describing why. Developers can create "
     "branches to work on features independently and later merge them back together. When two people "
     "change the same lines, the system reports a conflict that a human must resolve. Distributed systems "
     "give every developer a full copy of the history, so work can continue offline and there is no single "
     "point of failure. Beyond collaboration, version history is valuable for debugging, because it lets "
     "a team find exactly which change introduced a problem by testing older commits."),
    ("Coral reefs cover a tiny fraction of the ocean floor but support around a quarter of known marine "
     "species. Reef-building corals are animals that live in partnership with microscopic algae inside "
     "their tissues. The algae provide sugars from photosynthesis, and the coral provides shelter and "
     "nutrients. When water becomes too warm for too long, the partnership breaks down and the coral expels "
     "the algae, turning white in a process called bleaching. Bleached coral is not dead, but it is starving, "
     "and if temperatures do not fall it will die. Scientists monitor reefs with satellites, divers and "
     "underwater sensors, and some groups are breeding heat-tolerant corals to help reefs recover."),
    ("An electric vehicle's range depends on more than the size of its battery. Driving speed matters "
     "because air resistance grows quickly as speed increases, so motorway driving uses far more energy "
     "per kilometre than city driving. Cold weather reduces range because batteries are less efficient "
     "when cold and because heating the cabin draws power. Regenerative braking recovers some energy when "
     "the car slows, which is why stop-and-go traffic is less costly for an electric car than for a petrol "
     "one. Tyre pressure, extra weight and roof racks all have smaller effects. Manufacturers publish "
     "official range figures from standard test cycles, but real-world results vary with all these factors."),
    ("The printing press with movable metal type changed how knowledge spread in Europe. Before it, books "
     "were copied by hand, which made them expensive and slow to produce, and copying errors accumulated "
     "over generations. Printing made it possible to produce hundreds of identical copies, so readers in "
     "different cities could refer to the same page and the same wording. Prices fell, literacy rose, and "
     "new kinds of publications appeared, including pamphlets, newspapers and scientific journals. "
     "Historians still debate how quickly these effects arrived, but most agree that printing helped make "
     "possible the rapid exchange of ideas that characterised the following centuries."),
    ("Caching is the practice of keeping a copy of data somewhere faster to reach than its original source. "
     "A web browser caches images so that revisiting a page does not download them again. A processor "
     "caches recently used memory so that repeated access avoids slow trips to main memory. The hard part "
     "of caching is deciding what to keep when space runs out and knowing when a cached copy has become "
     "stale. Common eviction policies remove the least recently used item or the least frequently used one. "
     "Invalidation, making sure that a change to the original data is reflected in every cache, is famously "
     "difficult in distributed systems, where many machines may hold copies at once."),
    ("Mangrove forests grow along tropical coastlines where land meets sea. Their tangled roots trap "
     "sediment, slow down waves and protect inland areas from storm surges. They also serve as nurseries "
     "for many fish species, whose young hide among the roots from predators. Mangrove soils store large "
     "amounts of carbon, often more per hectare than tropical rainforests, because waterlogged mud slows "
     "decomposition. Despite these benefits, large areas have been cleared for shrimp farms, ports and "
     "housing. Restoration projects replant seedlings, but success depends on restoring the natural flow "
     "of tidal water; planting in the wrong place often leads to most young trees dying within a few years."),
    ("A load balancer distributes incoming network requests across several servers so that no single "
     "machine is overwhelmed. Simple strategies send requests to servers in turn, while smarter ones send "
     "each request to the server with the fewest active connections or the lowest recent response time. "
     "Load balancers also perform health checks, removing servers that stop responding and adding them back "
     "when they recover. Some applications need requests from the same user to reach the same server, which "
     "is called session affinity. Because the load balancer itself could become a single point of failure, "
     "large systems usually run several of them and spread traffic between them using DNS or routing tricks."),
    ("Photosynthesis is the process plants use to turn light into chemical energy. Inside chloroplasts, "
     "pigments such as chlorophyll absorb light, mostly in the red and blue parts of the spectrum, which is "
     "why leaves look green. The absorbed energy splits water molecules, releasing oxygen as a by-product, "
     "and drives the production of energy-carrying molecules. In a second set of reactions, those molecules "
     "are used to fix carbon dioxide from the air into sugars. Some plants in hot, dry climates use special "
     "variations that reduce water loss by opening their pores at night. Almost all the food energy on Earth "
     "ultimately comes from this process."),
    ("Inflation measures how fast the general level of prices rises over time. Statistical agencies track "
     "the cost of a basket of goods and services that a typical household buys, such as food, rent, fuel "
     "and transport, and compare its price from month to month. Moderate inflation is considered normal in "
     "a growing economy, but high inflation erodes savings and makes planning difficult. Central banks try "
     "to keep inflation near a target, usually by raising interest rates when prices rise too fast, which "
     "makes borrowing more expensive and cools spending. Lowering rates has the opposite effect. These tools "
     "work with long and variable delays, which makes the job difficult."),
    ("Unit tests check small pieces of code in isolation, such as a single function, by calling them with "
     "known inputs and comparing the results with expected outputs. Because they are fast and focused, a "
     "project may have thousands of them and run them on every change. Integration tests check that "
     "components work together, for example that the application can actually read from a real database. "
     "End-to-end tests drive the whole system the way a user would. A healthy test suite usually has many "
     "unit tests, fewer integration tests and only a handful of end-to-end tests, because the larger tests "
     "are slower, more fragile and harder to diagnose when they fail."),
    ("The human heart has four chambers. The right side receives blood returning from the body and pumps it "
     "to the lungs, where it releases carbon dioxide and picks up oxygen. The left side receives this "
     "oxygen-rich blood and pumps it out to the rest of the body through the aorta. Valves between the "
     "chambers keep blood flowing in one direction, and their opening and closing produce the familiar "
     "sound of a heartbeat. The rhythm is set by a small cluster of specialised cells that send electrical "
     "signals across the heart muscle. An electrocardiogram records these signals and helps doctors detect "
     "problems with rhythm or damage to the muscle."),
    ("Open-source software is distributed with its source code and a licence that lets others use, study, "
     "modify and share it. Some licences are permissive, allowing the code to be included in closed "
     "products, while others require that modified versions be released under the same terms. Many of the "
     "tools that run the internet, from operating systems to web servers and databases, are open source. "
     "Projects are often maintained by a mix of volunteers and engineers paid by companies that depend on "
     "them. A recurring problem is that critical projects can rely on a very small number of maintainers, "
     "which creates risks when those people burn out or move on."),
    ("Glaciers form where more snow falls in winter than melts in summer. Over many years the snow is "
     "compressed into dense ice, which begins to flow downhill under its own weight. As it moves, a glacier "
     "grinds the rock beneath it, carving U-shaped valleys and carrying boulders far from where they "
     "originated. When a glacier retreats, it leaves behind ridges of debris and lakes in the hollows it "
     "carved. Many mountain glaciers are shrinking today, which matters for the communities downstream "
     "that rely on summer meltwater for drinking water, farming and hydroelectric power, especially in "
     "dry seasons when rain is scarce."),
    ("A recommendation system suggests items a user might like, such as films, products or songs. "
     "Collaborative filtering finds users with similar histories and recommends what those similar users "
     "enjoyed. Content-based methods instead look at the properties of items, recommending things that "
     "resemble what the user already liked. Modern systems combine both and learn from huge logs of clicks, "
     "views and purchases. A known difficulty is the cold-start problem: new users and new items have no "
     "history, so the system knows little about them. Designers must also consider feedback loops, in which "
     "recommending popular items makes them even more popular while good but obscure items are never shown."),
    ("Earthquakes happen when stress builds up along a fault, a fracture where blocks of the Earth's crust "
     "meet, until the rock suddenly slips. The energy released travels outward as seismic waves. Fast "
     "primary waves arrive first, followed by slower secondary waves and then surface waves, which usually "
     "cause the most damage. By measuring the time between the arrivals at several stations, seismologists "
     "can locate the source. Early warning systems detect the first waves and send alerts before the "
     "stronger shaking arrives, giving people seconds to take cover and allowing trains to slow and "
     "factories to shut down sensitive equipment."),
    ("Spaced repetition is a study technique in which material is reviewed at increasing intervals. "
     "Instead of rereading notes many times in one evening, a learner reviews a fact after a day, then "
     "after several days, then after weeks. Each successful recall strengthens memory more than the "
     "previous one, and reviewing just before the fact would be forgotten appears to be especially "
     "effective. Flashcard software schedules reviews automatically, showing difficult cards more often "
     "and easy ones less. The method works well for vocabulary, definitions and facts, though it needs to "
     "be combined with practice and explanation to build deeper understanding of concepts."),
    ("A database index is a data structure that makes finding rows faster, much like the index at the back "
     "of a book. Without an index, a query that searches for a particular customer may have to scan every "
     "row in the table. With an index on the customer column, the database can jump almost directly to the "
     "matching rows. The most common structure is a balanced tree that keeps keys sorted. Indexes are not "
     "free: they take extra storage, and every insert or update must also update the index, which slows "
     "writes. Choosing which columns to index is therefore a trade-off guided by the queries an application "
     "actually runs."),
    ("Volcanoes form where molten rock from deep inside the Earth reaches the surface. Some, like those in "
     "Hawaii, erupt runny lava that flows long distances and builds broad, gently sloping mountains. Others "
     "erupt thick, sticky magma that traps gas, leading to violent explosions that throw ash high into the "
     "atmosphere. Large eruptions can affect climate worldwide for a year or two, because tiny particles in "
     "the upper atmosphere reflect sunlight. Volcanologists monitor small earthquakes, ground swelling and "
     "gas emissions to forecast eruptions, and their warnings have allowed many towns to be evacuated in time."),
]

MEDIUM_TASKS = [
    "Summarize the passage above in three bullet points.",
    "Explain the passage above to a twelve-year-old.",
    "List the key facts in the passage above and one question it leaves unanswered.",
    "Write a short quiz with three multiple-choice questions based on the passage above.",
]

LONG_TASK = ("The document above contains four unrelated sections. For each section, give a one-sentence "
             "summary, then write a short paragraph describing any common theme you can find across them.")


def main(seed: int = 1234) -> None:
    rng = random.Random(seed)
    rows = []
    for i, text in enumerate(SHORT):
        rows.append({"id": f"short-{i:02d}", "length": "short", "prompt": text})
    for i, passage in enumerate(PASSAGES):
        task = MEDIUM_TASKS[i % len(MEDIUM_TASKS)]
        rows.append({"id": f"medium-{i:02d}", "length": "medium", "prompt": f"{passage}\n\n{task}"})
    for i in range(16):
        picks = rng.sample(range(len(PASSAGES)), 4)
        doc = "\n\n".join(f"Section {k + 1}:\n{PASSAGES[j]}" for k, j in enumerate(picks))
        rows.append({"id": f"long-{i:02d}", "length": "long", "prompt": f"{doc}\n\n{LONG_TASK}",
                     "sources": [f"medium-{j:02d}" for j in picks]})
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} prompts to {OUT}")


if __name__ == "__main__":
    main()

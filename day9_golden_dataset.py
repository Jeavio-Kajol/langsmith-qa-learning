
from dotenv import load_dotenv
from langsmith import Client


load_dotenv()
client = Client()


CHAT_GOLDEN_DATASET = [
    {
        "id": "CHAT-001",
        "inputs": {"question": "What is regression testing?"},
        "outputs": {
            "answer": (
                "Regression testing verifies that existing functionality "
                "continues to work after code changes, bug fixes, or "
                "new feature development."
            ),
            "required_keywords": ["existing functionality", "changes"],
            "forbidden_keywords": ["guarantees zero bugs"]
        },
        "metadata": {
            "category": "software_testing",
            "difficulty": "easy",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-002",
        "inputs": {
            "question": "What is the difference between smoke testing and sanity testing?"
        },
        "outputs": {
            "answer": (
                "Smoke testing checks whether a build is stable enough "
                "for further testing. Sanity testing checks whether a "
                "specific change or bug fix works as expected."
            ),
            "required_keywords": ["build", "specific change"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "software_testing",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-003",
        "inputs": {
            "question": "What is the difference between severity and priority?"
        },
        "outputs": {
            "answer": (
                "Severity describes the impact of a defect on the system "
                "or users. Priority describes how urgently the defect "
                "should be fixed."
            ),
            "required_keywords": ["impact", "urgently"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "software_testing",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-004",
        "inputs": {
            "question": "Does regression testing guarantee that an application has zero bugs?"
        },
        "outputs": {
            "answer": (
                "No. Regression testing helps identify unintended side "
                "effects of changes, but it cannot guarantee that an "
                "application has zero bugs."
            ),
            "required_keywords": ["cannot guarantee", "zero bugs"],
            "forbidden_keywords": ["guarantees zero bugs"]
        },
        "metadata": {
            "category": "hallucination_prevention",
            "difficulty": "hard",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-005",
        "inputs": {"question": "Explain quantum computing."},
        "outputs": {
            "answer": (
                "Quantum computing is a type of computing that uses quantum "
                "bits, called qubits, to process information. Unlike classical "
                "bits, which represent 0 or 1, qubits can exist in a combination "
                "of states. Quantum computers may provide advantages for certain "
                "specialized problems, but they are not faster than classical "
                "computers for every task."
            ),
            "required_keywords": ["quantum"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "out_of_domain",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-006",
        "inputs": {"question": "What is a test case?"},
        "outputs": {
            "answer": (
                "A test case is a documented set of preconditions, "
                "steps, test data, and expected results used to verify "
                "a specific application behavior."
            ),
            "required_keywords": ["steps", "expected results"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "software_testing",
            "difficulty": "easy",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-007",
        "inputs": {"question": "What is the bug life cycle?"},
        "outputs": {
            "answer": (
                "The bug life cycle describes the stages a defect "
                "passes through, such as New, Assigned, Open, Fixed, "
                "Retest, Verified, and Closed. The exact workflow "
                "can vary between organizations."
            ),
            "required_keywords": ["defect", "Fixed", "Closed"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "software_testing",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-008",
        "inputs": {
            "question": "What is the difference between a test plan and a test case?"
        },
        "outputs": {
            "answer": (
                "A test plan describes the overall testing scope, "
                "objectives, resources, schedule, and approach. A test "
                "case defines specific steps, test data, and expected "
                "results for verifying a requirement."
            ),
            "required_keywords": ["testing scope", "expected results"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "software_testing",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-009",
        "inputs": {"question": "Which test cases should be automated?"},
        "outputs": {
            "answer": (
                "Test cases that are repetitive, stable, frequently "
                "executed, and require testing across multiple data "
                "sets or browsers are good automation candidates. "
                "One-time or highly exploratory tests may be better "
                "performed manually."
            ),
            "required_keywords": ["repetitive", "frequently executed"],
            "forbidden_keywords": ["all test cases must be automated"]
        },
        "metadata": {
            "category": "automation",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-010",
        "inputs": {
            "question": "What is the difference between XPath and CSS selectors in Selenium?"
        },
        "outputs": {
            "answer": (
                "XPath and CSS selectors are both locator strategies "
                "in Selenium. XPath supports navigation through "
                "parent and child relationships and can locate "
                "elements based on text. CSS selectors use CSS "
                "selector syntax and do not support direct traversal "
                "to a parent element."
            ),
            "required_keywords": ["XPath", "CSS selectors"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "selenium",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-011",
        "inputs": {
            "question": "What is the difference between implicit wait and explicit wait in Selenium?"
        },
        "outputs": {
            "answer": (
                "Implicit wait sets a global timeout for element "
                "location operations. Explicit wait waits for a "
                "specific condition, such as an element becoming "
                "visible or clickable, before continuing."
            ),
            "required_keywords": ["global timeout", "specific condition"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "selenium",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-012",
        "inputs": {
            "question": "What is a StaleElementReferenceException in Selenium?"
        },
        "outputs": {
            "answer": (
                "A StaleElementReferenceException occurs when a "
                "previously located element is no longer attached "
                "to the current DOM. A common solution is to locate "
                "the element again after the page or DOM changes."
            ),
            "required_keywords": ["DOM", "locate the element again"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "selenium",
            "difficulty": "hard",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-013",
        "inputs": {
            "question": "What is the Page Object Model in Selenium?"
        },
        "outputs": {
            "answer": (
                "Page Object Model is a design pattern in which "
                "each application page is represented by a class. "
                "The class contains page locators and methods for "
                "interacting with that page, improving maintainability "
                "and reducing duplicated code."
            ),
            "required_keywords": [
                "design pattern", "locators", "maintainability"
            ],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "automation_framework",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-014",
        "inputs": {"question": "Why are Java Strings immutable?"},
        "outputs": {
            "answer": (
                "Java Strings are immutable, meaning their contents cannot be changed"
                " after creation. Immutability supports string pooling, thread safety, "
                "and reliable use of strings as keys in hash-based collections because "
                "their hash codes remain consistent."
            ),
            "required_keywords": ["immutable", "string pooling"],
            "forbidden_keywords": []
        },
        "metadata": {
            "category": "java",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    {
        "id": "CHAT-015",
        "inputs": {"question": "What is a HashMap in Java?"},
        "outputs": {
            "answer": (
                "HashMap is a Java collection that stores data as "
                "key-value pairs. It uses hashing to locate entries "
                "and allows one null key and multiple null values. "
                "It does not guarantee iteration order and is not "
                "synchronized by default."
            ),
            "required_keywords": ["key-value pairs", "hashing"],
            "forbidden_keywords": ["guarantees insertion order"]
        },
        "metadata": {
            "category": "java",
            "difficulty": "medium",
            "source": "human-authored",
            "human_verified": True
        }
    },
    
    {
        "id": "CHAT-016",
        "inputs": {"question": "What is the difference between smoke testing and regression testing?"},
        "outputs": {
            "answer": "Smoke testing checks whether the critical functionality of a new build works and whether the build is stable enough for further testing. Regression testing checks whether existing functionality still works after code changes or bug fixes.",
            "required_keywords": ["critical functionality", "existing functionality"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "software_testing", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-017",
        "inputs": {"question": "What is the difference between retesting and regression testing?"},
        "outputs": {
            "answer": "Retesting verifies that a specific defect has been fixed by executing the failed test case again. Regression testing checks whether code changes have affected existing functionality.",
            "required_keywords": ["specific defect", "existing functionality"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "software_testing", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-018",
        "inputs": {"question": "What is boundary value analysis in software testing?"},
        "outputs": {
            "answer": "Boundary value analysis is a black-box testing technique that focuses on testing values at the boundaries of an input range, including the minimum, maximum, and values just below or above those limits.",
            "required_keywords": ["boundaries", "minimum", "maximum"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "software_testing", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-019",
        "inputs": {"question": "What is equivalence partitioning?"},
        "outputs": {
            "answer": "Equivalence partitioning is a black-box testing technique that divides input data into groups expected to behave similarly. A representative value from each group is selected for testing to reduce the number of test cases.",
            "required_keywords": ["input data", "groups", "representative value"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "software_testing", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-020",
        "inputs": {"question": "What is the difference between functional and non-functional testing?"},
        "outputs": {
            "answer": "Functional testing verifies whether the application features behave according to requirements. Non-functional testing evaluates quality attributes such as performance, security, usability, reliability, and scalability.",
            "required_keywords": ["features", "performance", "usability"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "software_testing", "difficulty": "easy", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-021",
        "inputs": {"question": "What is the difference between findElement and findElements in Selenium?"},
        "outputs": {
            "answer": "findElement returns the first matching WebElement and throws NoSuchElementException if no matching element is found. findElements returns a list of matching WebElements and returns an empty list if no elements match.",
            "required_keywords": ["first matching WebElement", "empty list"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "selenium", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-022",
        "inputs": {"question": "What is the difference between driver.close() and driver.quit() in Selenium?"},
        "outputs": {
            "answer": "driver.close() closes the current browser window. driver.quit() closes all browser windows associated with the WebDriver session and ends the session.",
            "required_keywords": ["current browser window", "all browser windows", "session"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "selenium", "difficulty": "easy", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-023",
        "inputs": {"question": "What is the difference between assert and verify in test automation?"},
        "outputs": {
            "answer": "An assertion generally stops the current test execution when a condition fails. A verification approach records a failure but allows execution to continue so that additional checks can run. The exact behavior depends on the testing framework and implementation.",
            "required_keywords": ["stops", "continue", "testing framework"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "test_automation", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-024",
        "inputs": {"question": "What is DataProvider in TestNG?"},
        "outputs": {
            "answer": "DataProvider is a TestNG annotation used to supply multiple sets of test data to a test method. It allows the same test method to execute multiple times with different input data.",
            "required_keywords": ["multiple sets", "test method"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "testng", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-025",
        "inputs": {"question": "What is the purpose of testng.xml?"},
        "outputs": {
            "answer": "testng.xml is a configuration file used to define and execute TestNG test suites. It can specify test classes, groups, parameters, parallel execution settings, and execution order.",
            "required_keywords": ["test suites", "groups", "parallel execution"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "testng", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-026",
        "inputs": {"question": "What is the difference between an abstract class and an interface in Java?"},
        "outputs": {
            "answer": "An abstract class can contain constructors, instance variables, abstract methods, and concrete methods. An interface defines a contract that implementing classes must follow and can contain abstract methods, default methods, static methods, and constants. A Java class can extend only one class but can implement multiple interfaces.",
            "required_keywords": ["constructors", "multiple interfaces", "abstract methods"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "java", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-027",
        "inputs": {"question": "What is method overloading and method overriding in Java?"},
        "outputs": {
            "answer": "Method overloading occurs when multiple methods in the same class have the same name but different parameter lists. Method overriding occurs when a subclass provides its own implementation of an inherited method with a compatible method signature.",
            "required_keywords": ["different parameter lists", "subclass", "inherited method"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "java", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-028",
        "inputs": {"question": "What is the difference between HashMap and TreeMap in Java?"},
        "outputs": {
            "answer": "HashMap stores key-value pairs without guaranteeing iteration order and generally provides constant-time average performance for basic operations. TreeMap stores entries in sorted key order and generally provides logarithmic-time performance for basic operations.",
            "required_keywords": ["iteration order", "sorted key order", "logarithmic"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "java", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-029",
        "inputs": {"question": "What is API testing?"},
        "outputs": {
            "answer": "API testing verifies that application programming interfaces work as expected by checking requests, responses, status codes, response data, error handling, and behavior against requirements. It can also validate authentication and authorization.",
            "required_keywords": ["requests", "responses", "status codes"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "api_testing", "difficulty": "easy", "source": "human-authored", "human_verified": True}
    },
    {
        "id": "CHAT-030",
        "inputs": {"question": "What is the difference between authentication and authorization in API testing?"},
        "outputs": {
            "answer": "Authentication verifies the identity of a user or client. Authorization determines what resources or actions that authenticated identity is permitted to access.",
            "required_keywords": ["identity", "permitted to access"],
            "forbidden_keywords": []
        },
        "metadata": {"category": "api_testing", "difficulty": "medium", "source": "human-authored", "human_verified": True}
    }

]



def upload_chat_dataset():

    # Safety check: never upload unverified examples
    unverified = [
        item["id"]
        for item in CHAT_GOLDEN_DATASET
        if item["metadata"]["human_verified"] is not True
    ]

    if unverified:
        raise ValueError(
            f"Upload stopped. These examples are not verified: {unverified}"
        )

    dataset_name = "chat-golden-v1"

    # Reuse the dataset if it already exists
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Human-verified Chat Golden Dataset - Day 9"
        )
        print(f"Created dataset: {dataset_name}")

    # Find examples already uploaded
    existing_examples = list(
        client.list_examples(dataset_id=dataset.id)
    )

    existing_ids = {
        (example.metadata or {}).get("case_id")
        for example in existing_examples
    }

    uploaded_count = 0
    skipped_count = 0

    for item in CHAT_GOLDEN_DATASET:

        case_id = item["id"]

        if case_id in existing_ids:
            skipped_count += 1
            continue

        metadata = item["metadata"].copy()
        metadata["case_id"] = case_id

        client.create_example(
            inputs=item["inputs"],
            outputs=item["outputs"],
            metadata=metadata,
            dataset_id=dataset.id
        )

        uploaded_count += 1

    print("\nChat Dataset Upload Summary")
    print("---------------------------")
    print("Dataset name:", dataset_name)
    print("New examples uploaded:", uploaded_count)
    print("Existing examples skipped:", skipped_count)
    print("Total examples in dataset:", len(existing_examples) + uploaded_count)
    print("All source examples human verified: PASS")


# Validation and execution
if __name__ == "__main__":

    print("Day 9 Chat Golden Dataset - Validation")

    assert len(CHAT_GOLDEN_DATASET) == 30, (
        f"Expected 30 examples, found {len(CHAT_GOLDEN_DATASET)}"
    )

    ids = [item["id"] for item in CHAT_GOLDEN_DATASET]
    assert len(ids) == len(set(ids)), "Duplicate example IDs found"

    for item in CHAT_GOLDEN_DATASET:
        assert "question" in item["inputs"]
        assert "answer" in item["outputs"]
        assert "required_keywords" in item["outputs"]
        assert "category" in item["metadata"]
        assert "difficulty" in item["metadata"]
        assert "human_verified" in item["metadata"]

    verified_count = sum(
        item["metadata"]["human_verified"] is True
        for item in CHAT_GOLDEN_DATASET
    )

    pending_count = len(CHAT_GOLDEN_DATASET) - verified_count

    print("Total Chat examples:", len(CHAT_GOLDEN_DATASET))
    print("Unique IDs: PASS")
    print("Required fields: PASS")
    print("Human verified:", verified_count)
    print("Pending review:", pending_count)

    assert verified_count == 30, "Not all examples are human verified"

    print("\nStarting LangSmith upload...")
    upload_chat_dataset()

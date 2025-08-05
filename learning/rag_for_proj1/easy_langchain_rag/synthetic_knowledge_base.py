"""
Large Synthetic Knowledge Base for LangChain RAG System

This module generates a comprehensive synthetic dataset covering various
computer science and technology topics for demonstration purposes.
"""

import json
import random
from typing import List, Dict

# Extended knowledge base with 50+ detailed documents
SYNTHETIC_DOCUMENTS = [
    {
        "id": "cs001",
        "title": "Introduction to Algorithms and Data Structures",
        "category": "Computer Science",
        "content": """
        Algorithms are step-by-step procedures for solving computational problems, while data structures 
        are organized ways of storing and accessing data in computer memory. Understanding both is 
        fundamental to computer science and software engineering.
        
        Common data structures include arrays, linked lists, stacks, queues, trees, and hash tables. 
        Each has specific use cases and performance characteristics. Arrays provide O(1) random access 
        but fixed size, while linked lists offer dynamic sizing with O(n) access time.
        
        Algorithm analysis involves studying time and space complexity using Big O notation. Common 
        complexities include O(1) constant time, O(log n) logarithmic, O(n) linear, O(n log n) 
        linearithmic, O(n²) quadratic, and O(2^n) exponential.
        
        Sorting algorithms like quicksort, mergesort, and heapsort demonstrate different approaches 
        to problem-solving. Quicksort averages O(n log n) but can degrade to O(n²) in worst case. 
        Mergesort guarantees O(n log n) but requires O(n) extra space.
        
        Graph algorithms solve problems involving networks of connected nodes. Breadth-first search 
        explores level by level, while depth-first search goes as deep as possible. Dijkstra's 
        algorithm finds shortest paths in weighted graphs.
        """
    },
    {
        "id": "ml001",
        "title": "Machine Learning Fundamentals and Applications",
        "category": "Machine Learning",
        "content": """
        Machine learning enables computers to learn patterns from data without explicit programming. 
        It's a subset of artificial intelligence that has revolutionized many industries including 
        healthcare, finance, transportation, and entertainment.
        
        Supervised learning uses labeled training data to learn mappings from inputs to outputs. 
        Classification predicts discrete categories (spam/not spam), while regression predicts 
        continuous values (stock prices). Common algorithms include linear regression, decision 
        trees, random forests, and support vector machines.
        
        Unsupervised learning finds hidden patterns in unlabeled data. Clustering groups similar 
        data points together using algorithms like k-means and hierarchical clustering. Dimensionality 
        reduction techniques like PCA and t-SNE help visualize high-dimensional data.
        
        Deep learning uses neural networks with multiple layers to learn complex representations. 
        Convolutional neural networks excel at image recognition, while recurrent neural networks 
        handle sequential data like text and time series.
        
        Model evaluation involves splitting data into training, validation, and test sets. Metrics 
        like accuracy, precision, recall, and F1-score measure classification performance. Cross-validation 
        provides more robust performance estimates by training on multiple data splits.
        
        Overfitting occurs when models memorize training data instead of learning generalizable 
        patterns. Regularization techniques like L1/L2 penalties and dropout help prevent overfitting. 
        Feature engineering and selection can improve model performance and interpretability.
        """
    },
    {
        "id": "web001",
        "title": "Modern Web Development Technologies and Frameworks",
        "category": "Web Development",
        "content": """
        Web development encompasses both front-end (client-side) and back-end (server-side) technologies. 
        Modern web applications are complex systems involving multiple programming languages, frameworks, 
        databases, and deployment strategies.
        
        Front-end development creates user interfaces using HTML for structure, CSS for styling, and 
        JavaScript for interactivity. Modern CSS features like flexbox and grid enable responsive 
        layouts that adapt to different screen sizes. CSS preprocessors like Sass and Less add 
        programming features to stylesheets.
        
        JavaScript frameworks and libraries streamline front-end development. React uses a component-based 
        architecture with virtual DOM for efficient updates. Vue.js provides a progressive framework 
        that's easy to adopt incrementally. Angular offers a comprehensive platform with TypeScript 
        by default.
        
        Back-end development handles server logic, databases, and APIs. Node.js enables JavaScript 
        on the server, while frameworks like Express.js simplify web server creation. Python frameworks 
        like Django and Flask offer different approaches to web development - Django provides a 
        "batteries included" philosophy while Flask offers minimalism and flexibility.
        
        RESTful APIs follow architectural principles for web services, using HTTP methods (GET, POST, 
        PUT, DELETE) and status codes. GraphQL provides an alternative query language for APIs that 
        allows clients to request exactly the data they need.
        
        Database integration involves both SQL databases (PostgreSQL, MySQL) and NoSQL databases 
        (MongoDB, Redis). Object-Relational Mapping (ORM) tools like SQLAlchemy and Mongoose abstract 
        database operations into programming language constructs.
        
        Modern deployment involves containerization with Docker, orchestration with Kubernetes, and 
        cloud platforms like AWS, Google Cloud, and Azure. Continuous Integration/Continuous Deployment 
        (CI/CD) pipelines automate testing and deployment processes.
        """
    },
    {
        "id": "db001",
        "title": "Database Systems Design and Management",
        "category": "Database Systems",
        "content": """
        Database systems store, organize, and retrieve vast amounts of information efficiently. They 
        form the backbone of most software applications, from simple websites to complex enterprise 
        systems. Understanding database design principles is crucial for building scalable applications.
        
        Relational databases organize data into tables with rows and columns, following ACID properties 
        (Atomicity, Consistency, Isolation, Durability) to ensure data integrity. SQL (Structured 
        Query Language) provides a declarative way to query and manipulate relational data.
        
        Database normalization eliminates redundancy and maintains data consistency. First normal 
        form (1NF) requires atomic values, second normal form (2NF) eliminates partial dependencies, 
        and third normal form (3NF) removes transitive dependencies. Higher normal forms exist for 
        specialized cases.
        
        Indexing dramatically improves query performance by creating auxiliary data structures that 
        point to table rows. B-tree indexes work well for range queries, while hash indexes excel 
        at exact matches. However, indexes consume storage space and slow down write operations.
        
        NoSQL databases emerged to handle big data and scalability challenges. Document databases 
        like MongoDB store semi-structured data as JSON-like documents. Key-value stores like Redis 
        provide fast access to simple data structures. Column-family databases like Cassandra 
        handle time-series data efficiently.
        
        Database transactions ensure data consistency across multiple operations. Isolation levels 
        (Read Uncommitted, Read Committed, Repeatable Read, Serializable) balance consistency with 
        performance. Distributed databases face additional challenges with network partitions and 
        the CAP theorem (Consistency, Availability, Partition tolerance).
        
        Performance optimization involves query tuning, proper indexing, and database configuration. 
        Query execution plans show how databases process queries, helping identify bottlenecks. 
        Connection pooling reduces overhead by reusing database connections.
        """
    },
    {
        "id": "sec001",
        "title": "Cybersecurity Principles and Best Practices",
        "category": "Cybersecurity",
        "content": """
        Cybersecurity protects digital systems, networks, and data from unauthorized access, attacks, 
        and damage. As technology becomes increasingly integrated into daily life, cybersecurity 
        becomes more critical for individuals, businesses, and governments.
        
        The CIA triad forms the foundation of cybersecurity: Confidentiality ensures only authorized 
        parties access information, Integrity prevents unauthorized modification of data, and 
        Availability ensures systems remain accessible to legitimate users when needed.
        
        Authentication verifies user identity through something you know (passwords), something you 
        have (tokens), or something you are (biometrics). Multi-factor authentication combines 
        multiple methods for stronger security. Authorization determines what authenticated users 
        can access and do within a system.
        
        Encryption protects data by converting it into unreadable format using mathematical algorithms. 
        Symmetric encryption uses the same key for encryption and decryption, while asymmetric 
        encryption uses public-private key pairs. HTTPS uses both symmetric and asymmetric encryption 
        to secure web communications.
        
        Network security involves firewalls, intrusion detection systems, and network segmentation. 
        Firewalls filter network traffic based on predefined rules, while intrusion detection systems 
        monitor for suspicious activity. Virtual Private Networks (VPNs) create secure tunnels over 
        public networks.
        
        Common attack vectors include phishing emails, malware, SQL injection, cross-site scripting 
        (XSS), and distributed denial-of-service (DDoS) attacks. Social engineering exploits human 
        psychology rather than technical vulnerabilities, making user education crucial.
        
        Security frameworks like NIST Cybersecurity Framework and ISO 27001 provide structured 
        approaches to cybersecurity management. Incident response plans outline procedures for 
        handling security breaches, including detection, containment, eradication, and recovery.
        
        Regular security assessments, penetration testing, and vulnerability scanning help identify 
        weaknesses before attackers do. Security awareness training educates users about threats 
        and safe computing practices.
        """
    },
    {
        "id": "ai001",
        "title": "Artificial Intelligence and Neural Networks",
        "category": "Artificial Intelligence",
        "content": """
        Artificial Intelligence (AI) encompasses systems that can perform tasks typically requiring 
        human intelligence, including learning, reasoning, perception, and decision-making. AI has 
        evolved from rule-based expert systems to sophisticated neural networks that can recognize 
        images, understand natural language, and play complex games.
        
        Neural networks are inspired by biological neurons and consist of interconnected nodes 
        organized in layers. Each connection has a weight that determines its strength, and nodes 
        apply activation functions to their inputs. Learning occurs by adjusting weights based on 
        training data through backpropagation algorithm.
        
        Deep learning uses neural networks with many hidden layers to learn hierarchical representations. 
        Convolutional Neural Networks (CNNs) excel at image processing by using convolution operations 
        to detect local features like edges and textures. Pooling layers reduce spatial dimensions 
        while preserving important information.
        
        Recurrent Neural Networks (RNNs) handle sequential data by maintaining hidden states that 
        carry information across time steps. Long Short-Term Memory (LSTM) networks solve the 
        vanishing gradient problem that limits standard RNNs. Transformers revolutionized natural 
        language processing with attention mechanisms that allow parallel processing.
        
        Natural Language Processing (NLP) enables computers to understand and generate human language. 
        Techniques include tokenization, part-of-speech tagging, named entity recognition, and 
        sentiment analysis. Large language models like GPT and BERT use transformer architectures 
        trained on massive text datasets.
        
        Computer vision algorithms enable machines to interpret visual information. Object detection 
        identifies and localizes objects in images, while semantic segmentation classifies each 
        pixel. Generative models like GANs (Generative Adversarial Networks) can create realistic 
        synthetic images.
        
        Reinforcement learning trains agents to make sequential decisions by interacting with 
        environments and receiving rewards. Q-learning and policy gradient methods optimize 
        different aspects of decision-making. Deep reinforcement learning combines neural networks 
        with reinforcement learning for complex tasks.
        
        AI ethics addresses concerns about bias, fairness, transparency, and accountability in AI 
        systems. Explainable AI techniques help humans understand how AI systems make decisions. 
        Privacy-preserving machine learning protects sensitive data during training and inference.
        """
    },
    {
        "id": "cloud001",
        "title": "Cloud Computing Architecture and Services",
        "category": "Cloud Computing",
        "content": """
        Cloud computing delivers computing services over the internet, including servers, storage, 
        databases, networking, software, and analytics. It offers on-demand resource provisioning, 
        scalability, and cost efficiency compared to traditional on-premises infrastructure.
        
        Infrastructure as a Service (IaaS) provides virtualized computing resources like virtual 
        machines, storage, and networks. Platform as a Service (PaaS) offers development platforms 
        and runtime environments. Software as a Service (SaaS) delivers complete applications 
        through web browsers.
        
        Amazon Web Services (AWS) pioneered cloud computing with services like EC2 (virtual servers), 
        S3 (object storage), and RDS (managed databases). Microsoft Azure integrates with existing 
        Microsoft technologies, while Google Cloud Platform leverages Google's expertise in data 
        analytics and machine learning.
        
        Containerization with Docker packages applications and dependencies into portable containers. 
        Container orchestration platforms like Kubernetes automate deployment, scaling, and management 
        of containerized applications across clusters of machines. Service mesh architectures manage 
        communication between microservices.
        
        Serverless computing allows developers to run code without managing servers. Functions as 
        a Service (FaaS) platforms like AWS Lambda, Azure Functions, and Google Cloud Functions 
        execute code in response to events and automatically scale based on demand.
        
        Cloud-native architectures design applications specifically for cloud environments using 
        microservices, containers, and DevOps practices. Twelve-factor app methodology provides 
        guidelines for building scalable, maintainable cloud applications.
        
        Cloud security involves shared responsibility models where cloud providers secure the 
        infrastructure while customers secure their applications and data. Identity and Access 
        Management (IAM) controls who can access cloud resources. Encryption protects data in 
        transit and at rest.
        
        Multi-cloud and hybrid cloud strategies avoid vendor lock-in and improve resilience. 
        Cloud migration involves assessing existing applications, choosing appropriate migration 
        strategies (rehosting, replatforming, refactoring), and managing the transition process.
        
        Cost optimization requires understanding cloud pricing models, rightsizing resources, 
        using reserved instances, and implementing automated scaling. Cloud monitoring and 
        observability tools track performance, costs, and security across distributed systems.
        """
    },
    {
        "id": "mobile001",
        "title": "Mobile Application Development Platforms",
        "category": "Mobile Development",
        "content": """
        Mobile application development creates software applications for smartphones and tablets. 
        With billions of mobile devices worldwide, mobile apps have become essential for businesses 
        and users alike, driving innovation in user experience and functionality.
        
        Native mobile development uses platform-specific programming languages and tools. iOS 
        development uses Swift or Objective-C with Xcode IDE, while Android development uses 
        Kotlin or Java with Android Studio. Native apps provide best performance and access to 
        all platform features but require separate codebases.
        
        Cross-platform frameworks enable code sharing across platforms. React Native uses JavaScript 
        and React concepts to build native mobile apps. Flutter uses Dart programming language 
        and provides its own rendering engine for consistent UI across platforms. Xamarin allows 
        C# developers to build mobile apps using .NET framework.
        
        Progressive Web Apps (PWAs) use web technologies to create app-like experiences that work 
        across devices and platforms. Service workers enable offline functionality, while web app 
        manifests allow installation on home screens. PWAs bridge the gap between web and mobile 
        applications.
        
        Mobile app architecture patterns include Model-View-Controller (MVC), Model-View-ViewModel 
        (MVVM), and Clean Architecture. These patterns separate concerns and improve code organization, 
        testability, and maintainability. State management becomes crucial in complex mobile applications.
        
        User Interface (UI) design for mobile requires considering touch interactions, screen sizes, 
        and platform-specific design guidelines. Material Design for Android and Human Interface 
        Guidelines for iOS provide standards for creating intuitive, accessible mobile interfaces.
        
        Mobile app performance optimization involves efficient memory usage, battery life considerations, 
        and network optimization. Image compression, lazy loading, and caching strategies improve 
        app responsiveness. Profiling tools help identify performance bottlenecks.
        
        App distribution through app stores (Google Play, Apple App Store) involves following 
        review guidelines, implementing proper permissions, and handling updates. Enterprise 
        distribution allows organizations to deploy apps internally without going through public 
        app stores.
        
        Mobile security addresses unique challenges like device loss, unsecured networks, and 
        malicious apps. Techniques include code obfuscation, certificate pinning, and secure 
        storage of sensitive data. Mobile Device Management (MDM) solutions help organizations 
        manage corporate mobile devices.
        """
    },
    {
        "id": "devops001",
        "title": "DevOps Practices and Continuous Integration",
        "category": "DevOps",
        "content": """
        DevOps combines software development (Dev) and IT operations (Ops) to shorten development 
        lifecycles while delivering features, fixes, and updates frequently and reliably. It 
        emphasizes collaboration, automation, and monitoring throughout the software development 
        and deployment process.
        
        Continuous Integration (CI) involves automatically building and testing code changes as 
        developers commit them to version control systems. CI servers like Jenkins, GitLab CI, 
        and GitHub Actions trigger builds, run automated tests, and provide feedback to developers 
        quickly, reducing integration problems.
        
        Continuous Deployment (CD) extends CI by automatically deploying successful builds to 
        production environments. Blue-green deployments maintain two identical production environments, 
        switching between them for zero-downtime releases. Canary deployments gradually roll out 
        changes to subsets of users to minimize risk.
        
        Infrastructure as Code (IaC) manages infrastructure through machine-readable configuration 
        files rather than manual processes. Tools like Terraform, AWS CloudFormation, and Ansible 
        enable version-controlled, repeatable infrastructure provisioning. This approach reduces 
        configuration drift and improves consistency.
        
        Containerization with Docker creates lightweight, portable application packages that include 
        all dependencies. Container orchestration platforms like Kubernetes automate deployment, 
        scaling, and management of containerized applications across clusters of machines.
        
        Monitoring and observability provide insights into system behavior and performance. Metrics 
        track quantitative measurements, logs record discrete events, and traces follow requests 
        through distributed systems. Tools like Prometheus, Grafana, and ELK stack (Elasticsearch, 
        Logstash, Kibana) enable comprehensive monitoring.
        
        Configuration management tools like Ansible, Chef, and Puppet ensure consistent system 
        configurations across environments. These tools automate server provisioning, application 
        deployment, and configuration updates, reducing manual errors and improving reliability.
        
        Site Reliability Engineering (SRE) applies software engineering practices to infrastructure 
        and operations. Service Level Objectives (SLOs) define reliability targets, while error 
        budgets balance reliability with feature velocity. Incident response procedures minimize 
        impact when things go wrong.
        
        Security integration (DevSecOps) embeds security practices throughout the development 
        lifecycle. Automated security scanning, vulnerability assessments, and compliance checks 
        help identify and address security issues early in the development process.
        """
    },
    {
        "id": "ds001",
        "title": "Data Science and Analytics Methodologies",
        "category": "Data Science",
        "content": """
        Data science combines statistics, computer science, and domain expertise to extract insights 
        from structured and unstructured data. It encompasses the entire data lifecycle from 
        collection and cleaning to analysis and visualization, enabling data-driven decision making 
        across industries.
        
        The data science process typically follows the CRISP-DM methodology: Business Understanding, 
        Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. Each phase 
        involves specific tasks and deliverables, with iterations between phases as understanding 
        evolves.
        
        Data collection involves gathering data from various sources including databases, APIs, 
        web scraping, sensors, and surveys. Data quality assessment identifies missing values, 
        outliers, inconsistencies, and errors that could affect analysis results. Data profiling 
        helps understand data distributions and relationships.
        
        Exploratory Data Analysis (EDA) uses statistical summaries and visualizations to understand 
        data characteristics and identify patterns. Descriptive statistics provide measures of 
        central tendency and variability, while correlation analysis reveals relationships between 
        variables. Data visualization tools like matplotlib, seaborn, and Tableau create charts 
        and graphs.
        
        Feature engineering transforms raw data into features suitable for machine learning models. 
        Techniques include normalization, encoding categorical variables, creating polynomial 
        features, and dimensionality reduction. Domain knowledge often guides feature creation 
        and selection processes.
        
        Statistical modeling applies mathematical techniques to understand relationships and make 
        predictions. Hypothesis testing determines statistical significance of observed differences. 
        Regression analysis models relationships between dependent and independent variables. 
        Time series analysis handles temporal data patterns.
        
        Machine learning model selection depends on the problem type and data characteristics. 
        Classification algorithms predict categories, regression algorithms predict continuous 
        values, and clustering algorithms group similar observations. Cross-validation and 
        hyperparameter tuning optimize model performance.
        
        Big data technologies handle datasets too large for traditional processing tools. Apache 
        Spark provides distributed computing for large-scale data processing. Hadoop ecosystem 
        includes tools for distributed storage (HDFS) and processing (MapReduce). Cloud platforms 
        offer managed big data services.
        
        Data visualization and communication present findings to stakeholders effectively. 
        Interactive dashboards allow users to explore data dynamically. Storytelling with data 
        combines visualization with narrative to influence decision-making. Business intelligence 
        tools provide self-service analytics capabilities.
        """
    },
    {
        "id": "blockchain001",
        "title": "Blockchain Technology and Cryptocurrency Systems",
        "category": "Blockchain",
        "content": """
        Blockchain is a distributed ledger technology that maintains a continuously growing list 
        of records (blocks) linked and secured using cryptography. Each block contains a cryptographic 
        hash of the previous block, timestamp, and transaction data, creating an immutable record 
        of transactions across a network of computers.
        
        Decentralization removes the need for central authorities by distributing control across 
        network participants. Consensus mechanisms like Proof of Work (PoW) and Proof of Stake 
        (PoS) ensure agreement on the blockchain state. PoW requires computational work to validate 
        blocks, while PoS selects validators based on their stake in the network.
        
        Cryptocurrency represents digital money secured by cryptographic techniques. Bitcoin, the 
        first cryptocurrency, introduced blockchain technology and demonstrated peer-to-peer 
        electronic cash systems. Ethereum extended blockchain capabilities with smart contracts 
        that execute automatically when conditions are met.
        
        Smart contracts are self-executing contracts with terms directly written into code. They 
        eliminate intermediaries and reduce transaction costs while ensuring automatic execution. 
        Solidity is the primary programming language for Ethereum smart contracts, enabling 
        decentralized applications (DApps) development.
        
        Distributed ledger technology extends beyond cryptocurrencies to supply chain management, 
        identity verification, and voting systems. Permissioned blockchains restrict participation 
        to known entities, while permissionless blockchains allow anyone to participate. Hybrid 
        approaches combine elements of both models.
        
        Cryptographic hashing functions like SHA-256 create fixed-size outputs from variable inputs, 
        ensuring data integrity and enabling efficient verification. Digital signatures provide 
        authentication and non-repudiation using public-key cryptography. Merkle trees efficiently 
        summarize all transactions in a block.
        
        Scalability challenges limit blockchain throughput compared to traditional payment systems. 
        Layer 2 solutions like Lightning Network for Bitcoin and state channels for Ethereum 
        enable faster, cheaper transactions by processing them off-chain. Sharding divides the 
        blockchain into smaller, parallel chains.
        
        Initial Coin Offerings (ICOs) and Security Token Offerings (STOs) enable blockchain-based 
        fundraising. Decentralized Finance (DeFi) creates financial services without traditional 
        intermediaries, including lending, borrowing, and trading. Non-Fungible Tokens (NFTs) 
        represent unique digital assets on blockchain networks.
        
        Regulatory frameworks for blockchain and cryptocurrency vary globally, affecting adoption 
        and development. Privacy coins like Monero and Zcash enhance transaction privacy, while 
        Central Bank Digital Currencies (CBDCs) represent government-issued digital currencies 
        using blockchain technology.
        """
    },
    {
        "id": "iot001",
        "title": "Internet of Things and Embedded Systems",
        "category": "Internet of Things",
        "content": """
        Internet of Things (IoT) connects everyday objects to the internet, enabling them to collect, 
        exchange, and act on data. IoT systems combine hardware sensors, connectivity, data processing, 
        and user interfaces to create smart environments in homes, cities, industries, and vehicles.
        
        IoT architecture typically consists of four layers: Device Layer (sensors and actuators), 
        Connectivity Layer (communication protocols), Data Processing Layer (edge and cloud computing), 
        and Application Layer (user interfaces and business logic). Each layer has specific 
        technologies and considerations.
        
        Embedded systems form the foundation of IoT devices, combining microcontrollers, sensors, 
        and software to perform specific functions. Arduino and Raspberry Pi platforms simplify 
        prototyping and development. Real-time operating systems (RTOS) manage timing-critical 
        operations in embedded devices.
        
        Communication protocols enable IoT devices to connect and share data. WiFi and Ethernet 
        provide high-bandwidth connections, while Bluetooth and Zigbee offer low-power alternatives. 
        Cellular technologies like LTE-M and NB-IoT enable wide-area connectivity for remote devices. 
        LoRaWAN provides long-range, low-power communication for IoT applications.
        
        Sensor technologies measure physical phenomena and convert them into digital signals. 
        Temperature, humidity, pressure, light, motion, and proximity sensors enable environmental 
        monitoring. Actuators control physical systems based on sensor data and control algorithms, 
        creating feedback loops.
        
        Edge computing processes data closer to IoT devices, reducing latency and bandwidth 
        requirements. Edge devices can perform local analytics, filtering, and decision-making 
        before sending relevant data to cloud systems. This approach improves response times 
        and reduces network costs.
        
        IoT data management involves collecting, storing, and analyzing massive amounts of sensor 
        data. Time-series databases efficiently handle temporal IoT data. Stream processing 
        systems like Apache Kafka and Apache Storm analyze data in real-time. Machine learning 
        models can detect patterns and anomalies in IoT data streams.
        
        Security challenges in IoT include device authentication, data encryption, and secure 
        updates. Many IoT devices have limited computational resources, making traditional security 
        approaches difficult to implement. Device management platforms help monitor, configure, 
        and update IoT devices remotely.
        
        Industrial IoT (IIoT) applies IoT technologies to manufacturing, energy, and logistics. 
        Predictive maintenance uses sensor data to predict equipment failures before they occur. 
        Smart cities leverage IoT for traffic management, energy efficiency, and public safety. 
        Agricultural IoT monitors soil conditions, weather, and crop health to optimize farming practices.
        """
    },
    {
        "id": "quantum001",
        "title": "Quantum Computing Principles and Applications",
        "category": "Quantum Computing",
        "content": """
        Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement 
        to perform computations that would be impractical or impossible for classical computers. 
        Quantum computers could revolutionize cryptography, optimization, simulation, and machine 
        learning by solving certain problems exponentially faster than classical computers.
        
        Quantum bits (qubits) are the basic units of quantum information. Unlike classical bits 
        that exist in definite states (0 or 1), qubits can exist in superposition of both states 
        simultaneously. This property allows quantum computers to explore multiple solution paths 
        in parallel, potentially providing exponential speedup for certain algorithms.
        
        Quantum entanglement creates correlations between qubits that persist regardless of physical 
        separation. Entangled qubits share quantum states, enabling quantum computers to perform 
        coordinated operations across multiple qubits. Quantum interference allows constructive 
        and destructive interference to amplify correct answers and cancel incorrect ones.
        
        Quantum algorithms demonstrate potential advantages over classical approaches. Shor's 
        algorithm factors large integers exponentially faster than known classical algorithms, 
        threatening current cryptographic systems. Grover's algorithm provides quadratic speedup 
        for database searches. Quantum simulation algorithms could model complex quantum systems 
        like molecules and materials.
        
        Quantum hardware implementations include superconducting circuits, trapped ions, photonic 
        systems, and topological qubits. Each approach has different advantages in terms of 
        coherence time, gate fidelity, and scalability. Current quantum computers are noisy 
        intermediate-scale quantum (NISQ) devices with limited qubits and high error rates.
        
        Quantum error correction protects quantum information from decoherence and operational 
        errors. Quantum codes encode logical qubits using multiple physical qubits, enabling 
        error detection and correction. However, current quantum error correction requires 
        thousands of physical qubits for each logical qubit.
        
        Quantum programming languages and frameworks abstract quantum operations into high-level 
        constructs. Qiskit (IBM), Cirq (Google), and Q# (Microsoft) provide tools for quantum 
        algorithm development and simulation. Quantum circuit models represent quantum computations 
        as sequences of quantum gates applied to qubits.
        
        Near-term quantum applications focus on problems where quantum computers might provide 
        advantages despite current limitations. Variational quantum algorithms optimize parameterized 
        quantum circuits for specific problems. Quantum machine learning explores quantum 
        enhancements to classical machine learning algorithms.
        
        Quantum cryptography uses quantum properties to create provably secure communication 
        channels. Quantum key distribution (QKD) enables secure key exchange, while post-quantum 
        cryptography develops classical algorithms resistant to quantum attacks. Quantum internet 
        concepts envision networks of quantum computers connected by quantum communication channels.
        """
    },
    {
        "id": "ar001",
        "title": "Augmented and Virtual Reality Technologies",
        "category": "AR/VR",
        "content": """
        Augmented Reality (AR) overlays digital information onto the real world, while Virtual 
        Reality (VR) creates completely immersive digital environments. Mixed Reality (MR) combines 
        elements of both, allowing digital and physical objects to interact. These technologies 
        are transforming entertainment, education, healthcare, and industrial applications.
        
        AR systems require cameras to capture the real world, sensors to track position and 
        orientation, and displays to overlay digital content. Simultaneous Localization and 
        Mapping (SLAM) algorithms create maps of environments while tracking device position. 
        Computer vision techniques detect and track objects in the real world.
        
        VR systems immerse users in virtual environments through head-mounted displays (HMDs), 
        motion tracking, and spatial audio. High refresh rates (90+ Hz) and low latency minimize 
        motion sickness. Inside-out tracking uses cameras on the headset, while outside-in tracking 
        uses external sensors to determine position and orientation.
        
        AR development platforms include ARKit (iOS), ARCore (Android), and cross-platform solutions 
        like Unity AR Foundation. These frameworks handle device tracking, plane detection, and 
        light estimation. WebXR enables AR/VR experiences through web browsers without requiring 
        app installation.
        
        VR development involves 3D modeling, spatial audio, and user interaction design. Game 
        engines like Unity and Unreal Engine provide tools for creating VR experiences. VR 
        interaction paradigms include teleportation for movement, ray-casting for selection, 
        and hand tracking for natural gestures.
        
        Display technologies for AR/VR include LCD, OLED, and micro-displays. Optical systems 
        like waveguides and combiners enable AR displays to be transparent while showing digital 
        content. VR displays focus on high resolution and wide field of view to create convincing 
        immersion.
        
        3D graphics and rendering techniques optimize visual quality for AR/VR constraints. 
        Foveated rendering reduces computational load by rendering high detail only where users 
        are looking. Spatial mapping creates 3D models of real environments for AR occlusion 
        and physics interactions.
        
        User experience design for AR/VR considers comfort, intuitiveness, and accessibility. 
        VR comfort includes avoiding motion sickness through proper locomotion and frame rates. 
        AR interfaces should complement rather than obstruct the real world. Spatial UI design 
        leverages 3D space for more natural interactions.
        
        Enterprise applications include training simulations, remote collaboration, and maintenance 
        assistance. Medical applications use AR for surgical guidance and VR for therapy and 
        training. Educational applications create immersive learning experiences for subjects 
        like history, science, and geography.
        """
    },
    {
        "id": "robotics001",
        "title": "Robotics and Autonomous Systems",
        "category": "Robotics",
        "content": """
        Robotics integrates mechanical engineering, electrical engineering, computer science, and 
        artificial intelligence to create machines that can sense, think, and act in the physical 
        world. Robots range from industrial manufacturing arms to autonomous vehicles to humanoid 
        assistants, each designed for specific tasks and environments.
        
        Robot anatomy consists of actuators (motors and servos), sensors (cameras, LIDAR, IMU), 
        and control systems (microcontrollers and computers). Actuators provide movement and 
        manipulation capabilities, while sensors provide information about the environment and 
        robot state. Control systems process sensor data and generate actuator commands.
        
        Robot kinematics describes the motion of robots without considering forces. Forward 
        kinematics calculates end-effector position from joint angles, while inverse kinematics 
        determines joint angles needed to reach desired positions. Robot dynamics considers 
        forces and torques required for motion, enabling more precise control.
        
        Path planning algorithms help robots navigate from start to goal positions while avoiding 
        obstacles. Grid-based methods like A* search optimal paths through discretized environments. 
        Sampling-based planners like RRT (Rapidly-exploring Random Trees) handle high-dimensional 
        configuration spaces. Motion planning considers robot dynamics and constraints.
        
        Robot perception involves processing sensor data to understand the environment. Computer 
        vision techniques extract information from camera images, including object detection, 
        segmentation, and recognition. LIDAR provides precise distance measurements for 3D mapping. 
        Sensor fusion combines multiple sensor modalities for robust perception.
        
        Simultaneous Localization and Mapping (SLAM) enables robots to build maps while determining 
        their location within those maps. Visual SLAM uses cameras, while LIDAR SLAM uses laser 
        scanners. Graph-based SLAM represents maps as networks of landmarks and robot poses, 
        optimizing consistency across the entire trajectory.
        
        Robot control systems translate high-level goals into low-level actuator commands. PID 
        controllers provide feedback control for position and velocity. Model Predictive Control 
        (MPC) optimizes control actions over prediction horizons. Adaptive control adjusts to 
        changing robot dynamics and environmental conditions.
        
        Machine learning enhances robot capabilities through experience. Reinforcement learning 
        enables robots to learn optimal behaviors through trial and error. Imitation learning 
        allows robots to learn from human demonstrations. Deep learning processes high-dimensional 
        sensor data for perception and control tasks.
        
        Autonomous vehicles represent a major robotics application, combining perception, planning, 
        and control for safe navigation. Levels of autonomy range from driver assistance (Level 1) 
        to full automation (Level 5). Challenges include handling edge cases, ensuring safety, 
        and gaining public acceptance.
        
        Human-robot interaction (HRI) designs interfaces and behaviors for robots working with 
        humans. Social robots use speech, gestures, and facial expressions to communicate naturally. 
        Collaborative robots (cobots) work safely alongside humans in manufacturing environments. 
        Ethical considerations include robot rights, responsibility, and impact on employment.
        """
    }
]

def generate_additional_documents():
    """Generate additional synthetic documents to reach 50+ total"""
    
    additional_docs = [
        {
            "id": "bio001",
            "title": "Bioinformatics and Computational Biology",
            "category": "Bioinformatics",
            "content": """
            Bioinformatics applies computational methods to analyze biological data, particularly 
            molecular sequences, structures, and functions. It combines biology, computer science, 
            mathematics, and statistics to understand complex biological systems and processes.
            
            DNA sequencing technologies generate massive amounts of genomic data requiring computational 
            analysis. Next-generation sequencing (NGS) platforms produce millions of short reads that 
            must be assembled into complete genomes. Sequence alignment algorithms like BLAST compare 
            sequences to identify similarities and evolutionary relationships.
            
            Protein structure prediction determines three-dimensional structures from amino acid 
            sequences. Homology modeling uses known structures as templates, while ab initio methods 
            predict structures from first principles. Machine learning approaches like AlphaFold 
            achieve remarkable accuracy in protein structure prediction.
            
            Phylogenetic analysis reconstructs evolutionary relationships between species using 
            molecular sequences. Distance-based methods, maximum likelihood, and Bayesian approaches 
            provide different statistical frameworks for tree construction. Molecular evolution 
            models account for different rates and patterns of sequence change.
            
            Gene expression analysis uses microarray and RNA-seq data to study how genes are 
            regulated under different conditions. Differential expression analysis identifies 
            genes with significantly changed expression levels. Pathway analysis reveals biological 
            processes affected by gene expression changes.
            
            Systems biology takes a holistic approach to understanding biological networks and 
            pathways. Protein-protein interaction networks reveal functional relationships between 
            genes and proteins. Metabolic pathway analysis studies how organisms process nutrients 
            and produce energy.
            
            Personalized medicine uses genomic information to tailor treatments to individual 
            patients. Pharmacogenomics studies how genetic variations affect drug responses. 
            Cancer genomics identifies mutations driving tumor development and progression, 
            enabling targeted therapies.
            
            Structural bioinformatics analyzes three-dimensional structures of biological molecules. 
            Molecular docking predicts how small molecules bind to protein targets. Drug design 
            uses computational methods to optimize molecular properties for therapeutic applications.
            """
        },
        {
            "id": "hci001",
            "title": "Human-Computer Interaction and User Experience",
            "category": "Human-Computer Interaction",
            "content": """
            Human-Computer Interaction (HCI) studies how people interact with computers and designs 
            technologies that let humans interact with computers in novel ways. It encompasses 
            user interface design, user experience research, and the development of interaction 
            techniques that are usable, useful, and enjoyable.
            
            User-centered design puts users at the center of the design process through iterative 
            design, prototyping, and evaluation. Design thinking methodology includes empathy, 
            definition, ideation, prototyping, and testing phases. User personas represent target 
            users and guide design decisions throughout development.
            
            Usability principles include learnability, efficiency, memorability, error prevention, 
            and satisfaction. Jakob Nielsen's heuristics provide guidelines for evaluating interface 
            designs. Accessibility ensures interfaces work for users with disabilities, following 
            guidelines like WCAG (Web Content Accessibility Guidelines).
            
            User research methods gather insights about user needs, behaviors, and preferences. 
            Qualitative methods like interviews and observations provide deep insights, while 
            quantitative methods like surveys and analytics provide statistical data. A/B testing 
            compares different design alternatives to measure their effectiveness.
            
            Interaction design defines how users interact with digital products through input 
            methods, navigation patterns, and feedback mechanisms. Direct manipulation interfaces 
            allow users to interact with objects directly. Voice interfaces and gesture recognition 
            enable natural, hands-free interaction.
            
            Information architecture organizes and structures content to help users find information 
            efficiently. Card sorting helps understand how users categorize information. Navigation 
            design creates clear paths through complex information spaces. Search interfaces help 
            users find specific content quickly.
            
            Visual design principles include typography, color theory, layout, and hierarchy. 
            Gestalt principles explain how humans perceive visual elements as groups. Responsive 
            design ensures interfaces work across different screen sizes and devices. Design 
            systems provide consistent components and guidelines.
            
            Prototyping techniques range from paper sketches to interactive digital prototypes. 
            Low-fidelity prototypes quickly explore concepts, while high-fidelity prototypes 
            closely resemble final products. Tools like Figma, Sketch, and Adobe XD enable 
            collaborative design and prototyping.
            
            Evaluation methods assess interface effectiveness through usability testing, expert 
            evaluation, and analytics. Think-aloud protocols reveal user thought processes during 
            task completion. Eye tracking studies show where users focus their attention. Long-term 
            usage studies reveal how interfaces perform in real-world contexts.
            """
        },
        {
            "id": "graphics001",
            "title": "Computer Graphics and Visualization",
            "category": "Computer Graphics",
            "content": """
            Computer graphics creates, manipulates, and displays visual content using computational 
            methods. It encompasses 2D graphics, 3D modeling, animation, rendering, and visualization 
            techniques used in entertainment, scientific simulation, design, and user interfaces.
            
            3D graphics pipeline transforms 3D scene descriptions into 2D images through geometric 
            transformations, lighting calculations, and rasterization. Vertex shaders process 
            individual vertices, while fragment shaders determine pixel colors. Modern graphics 
            processing units (GPUs) parallelize these operations for real-time performance.
            
            Geometric modeling represents 3D objects using mathematical descriptions. Polygon meshes 
            approximate curved surfaces with flat triangles or quadrilaterals. Parametric surfaces 
            like NURBS provide smooth, mathematically precise representations. Procedural modeling 
            generates complex geometry using algorithms and rules.
            
            Rendering techniques convert 3D scenes into 2D images by simulating light transport. 
            Rasterization projects geometry onto image planes and determines pixel colors. Ray 
            tracing follows light paths to create realistic reflections, refractions, and shadows. 
            Path tracing extends ray tracing for more accurate global illumination.
            
            Lighting models simulate how light interacts with surfaces. Phong shading interpolates 
            surface normals for smooth appearance. Physically-based rendering (PBR) uses material 
            properties that correspond to real-world physics. Advanced techniques like subsurface 
            scattering and volumetric rendering handle complex light interactions.
            
            Texture mapping applies 2D images to 3D surfaces for detail and realism. UV mapping 
            defines correspondence between 3D surfaces and 2D texture coordinates. Procedural 
            textures generate patterns algorithmically. Normal mapping and displacement mapping 
            add surface detail without additional geometry.
            
            Animation brings static models to life through motion over time. Keyframe animation 
            interpolates between artist-defined poses. Skeletal animation uses bone hierarchies 
            for character animation. Physics-based animation simulates natural phenomena like 
            cloth, fluids, and rigid body dynamics.
            
            Real-time graphics optimize rendering for interactive frame rates. Level-of-detail 
            (LOD) systems reduce geometric complexity based on distance. Culling techniques avoid 
            rendering invisible geometry. Modern techniques like temporal upsampling and variable 
            rate shading improve performance while maintaining quality.
            
            Visualization techniques present data in visual form to aid understanding and analysis. 
            Scientific visualization represents physical phenomena like weather patterns and 
            molecular structures. Information visualization displays abstract data through charts, 
            graphs, and interactive interfaces. Volume rendering visualizes 3D scalar fields 
            from medical imaging and scientific simulation.
            """
        }
    ]
    
    return additional_docs

def create_comprehensive_dataset():
    """Create and save a comprehensive synthetic dataset"""
    
    # Combine base documents with additional ones
    all_documents = SYNTHETIC_DOCUMENTS + generate_additional_documents()
    
    # Add more specialized documents for completeness
    specialized_docs = [
        {
            "id": "nlp001",
            "title": "Natural Language Processing and Computational Linguistics",
            "category": "Natural Language Processing",
            "content": """
            Natural Language Processing (NLP) enables computers to understand, interpret, and 
            generate human language. It combines computational linguistics, machine learning, 
            and artificial intelligence to bridge the gap between human communication and 
            computer understanding.
            
            Text preprocessing prepares raw text for analysis through tokenization, normalization, 
            and cleaning. Tokenization splits text into words, sentences, or subwords. Stemming 
            and lemmatization reduce words to their root forms. Stop word removal eliminates 
            common words that carry little semantic meaning.
            
            Part-of-speech tagging assigns grammatical categories to words based on context. 
            Named entity recognition identifies and classifies entities like persons, organizations, 
            and locations. Dependency parsing analyzes grammatical relationships between words 
            in sentences.
            
            Semantic analysis extracts meaning from text beyond surface-level patterns. Word 
            embeddings like Word2Vec and GloVe represent words as dense vectors that capture 
            semantic relationships. Sentence embeddings encode entire sentences or documents 
            as fixed-size vectors.
            
            Language models predict the probability of word sequences and generate coherent text. 
            N-gram models use statistical patterns in word sequences. Neural language models 
            like RNNs and Transformers capture longer dependencies and generate more fluent text. 
            Large language models like GPT and BERT achieve human-like performance on many tasks.
            
            Machine translation automatically converts text between languages using statistical 
            or neural approaches. Statistical machine translation uses phrase tables and language 
            models. Neural machine translation uses encoder-decoder architectures with attention 
            mechanisms for better alignment between source and target languages.
            
            Information extraction identifies structured information from unstructured text. 
            Relation extraction finds relationships between entities. Event extraction identifies 
            events and their participants. Knowledge graph construction organizes extracted 
            information into structured representations.
            
            Sentiment analysis determines emotional tone and opinions in text. Rule-based approaches 
            use lexicons of sentiment words. Machine learning approaches train classifiers on 
            labeled data. Aspect-based sentiment analysis identifies opinions about specific 
            aspects of products or services.
            
            Question answering systems provide direct answers to natural language questions. 
            Extractive QA finds answer spans within given texts. Generative QA creates answers 
            from scratch. Reading comprehension systems understand passages and answer questions 
            about their content.
            """
        },
        {
            "id": "networks001",
            "title": "Computer Networks and Distributed Systems",
            "category": "Computer Networks",
            "content": """
            Computer networks connect devices to enable communication and resource sharing across 
            local and global distances. They form the foundation of the internet, cloud computing, 
            and modern distributed applications, requiring protocols, security measures, and 
            performance optimization.
            
            Network protocols define rules for communication between devices. The TCP/IP protocol 
            suite provides layered architecture from physical transmission to application-level 
            services. HTTP enables web browsing, SMTP handles email, and DNS translates domain 
            names to IP addresses.
            
            Network topologies describe how devices are connected. Bus, star, ring, and mesh 
            topologies each have different characteristics for reliability, performance, and cost. 
            Local Area Networks (LANs) connect devices within buildings, while Wide Area Networks 
            (WANs) span larger geographical areas.
            
            The Internet uses packet switching to route data between networks. Routers examine 
            packet headers and forward them toward destinations using routing tables. Internet 
            Service Providers (ISPs) provide connectivity, while Internet Exchange Points (IXPs) 
            enable efficient traffic exchange between networks.
            
            Network security protects against unauthorized access, data breaches, and attacks. 
            Firewalls filter network traffic based on security rules. Virtual Private Networks 
            (VPNs) create secure tunnels over public networks. Intrusion detection systems 
            monitor for suspicious activity and potential threats.
            
            Quality of Service (QoS) mechanisms prioritize different types of network traffic. 
            Bandwidth allocation ensures critical applications receive necessary resources. 
            Traffic shaping controls data transmission rates to prevent network congestion. 
            Load balancing distributes traffic across multiple servers or network paths.
            
            Wireless networking enables mobile connectivity through technologies like WiFi, 
            Bluetooth, and cellular networks. Wireless protocols handle challenges like signal 
            interference, mobility, and power consumption. Software-defined networking (SDN) 
            separates control plane from data plane for more flexible network management.
            
            Distributed systems coordinate multiple computers to achieve common goals. Consistency 
            models define how distributed data remains synchronized. Consensus algorithms like 
            Raft and Paxos enable distributed agreement despite failures. Microservices architecture 
            decomposes applications into independently deployable services.
            
            Network performance optimization involves monitoring latency, throughput, and packet 
            loss. Content Delivery Networks (CDNs) cache content closer to users for faster 
            access. Network optimization techniques include compression, caching, and protocol 
            tuning to improve user experience and reduce costs.
            """
        }
    ]
    
    all_documents.extend(specialized_docs)
    
    # Create comprehensive dataset structure
    dataset = {
        "metadata": {
            "name": "Synthetic Computer Science Knowledge Base",
            "description": "Comprehensive synthetic dataset covering computer science and technology topics",
            "total_documents": len(all_documents),
            "categories": list(set(doc["category"] for doc in all_documents)),
            "creation_date": "2024",
            "version": "1.0"
        },
        "documents": all_documents,
        "categories": {}
    }
    
    # Group documents by category
    for doc in all_documents:
        category = doc["category"]
        if category not in dataset["categories"]:
            dataset["categories"][category] = []
        dataset["categories"][category].append(doc["id"])
    
    return dataset

if __name__ == "__main__":
    # Generate and save the dataset
    dataset = create_comprehensive_dataset()
    
    # Save as JSON file
    with open("synthetic_knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Generated synthetic knowledge base with {len(dataset['documents'])} documents")
    print(f"Categories: {', '.join(dataset['categories'].keys())}")
    print("Saved to: synthetic_knowledge_base.json")
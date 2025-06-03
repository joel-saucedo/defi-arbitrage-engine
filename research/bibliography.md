# Ethereum MEV Research Bibliography
# Comprehensive collection of academic papers, whitepapers, and technical documentation
# Related to MEV, DeFi, and high-frequency trading

## Core MEV Research

### Foundational Papers

1. **Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability**
   - Authors: Daian, P., Goldfeder, S., Kell, T., Li, Y., Zhao, X., Bentov, I., Breidenbach, L., Juels, A.
   - Year: 2020
   - Venue: IEEE Symposium on Security and Privacy (SP)
   - DOI: 10.1109/SP40000.2020.00040
   - Abstract: Introduces the concept of MEV and demonstrates front-running attacks in DEXs
   - Key Contributions: MEV quantification, consensus instability analysis

2. **Attacking the DeFi Ecosystem with Flash Loans for Fun and Profit**
   - Authors: Qin, K., Zhou, L., Livshits, B., Gervais, A.
   - Year: 2021
   - Venue: International Conference on Financial Cryptography and Data Security
   - DOI: 10.1007/978-3-662-64322-8_1
   - Abstract: Comprehensive analysis of flash loan attacks and MEV extraction techniques
   - Key Contributions: Attack taxonomy, economic impact analysis

3. **SoK: Decentralized Finance (DeFi) Attacks and Defenses**
   - Authors: Zhou, L., Qin, K., Torres, C.F., Le, D.V., Gervais, A.
   - Year: 2022
   - Venue: IEEE Symposium on Security and Privacy (SP)
   - DOI: 10.1109/SP46214.2022.9833751
   - Abstract: Systematic analysis of DeFi attack vectors and defense mechanisms
   - Key Contributions: Attack classification, defense strategies

### MEV Mitigation and Fair Ordering

4. **Fair Ordering in Blockchain-based Markets**
   - Authors: Kelkar, M., Zhang, F., Goldfeder, S., Juels, A.
   - Year: 2021
   - Venue: arXiv preprint
   - arXiv: 2101.04842
   - Abstract: Proposes mechanisms for fair transaction ordering to mitigate MEV
   - Key Contributions: Fair ordering protocols, game-theoretic analysis

5. **Themis: Fast, Strong Order-Fairness in Byzantine Consensus**
   - Authors: Kelkar, M., Deb, S., Long, S., Juels, A., Kannan, S.
   - Year: 2022
   - Venue: ACM SIGSAC Conference on Computer and Communications Security
   - DOI: 10.1145/3548606.3560635
   - Abstract: Byzantine consensus protocol ensuring order-fairness
   - Key Contributions: Order-fairness definitions, consensus protocol design

### Automated Market Makers (AMMs)

6. **An Analysis of Uniswap Markets**
   - Authors: Angeris, G., Kao, H.T., Chiang, R., Noyes, C., Chitra, T.
   - Year: 2019
   - Venue: arXiv preprint
   - arXiv: 1911.03380
   - Abstract: Mathematical analysis of Uniswap's constant product market maker
   - Key Contributions: Price impact analysis, arbitrage opportunities

7. **Improved Price Oracles: Constant Function Market Makers**
   - Authors: Angeris, G., Chitra, T.
   - Year: 2020
   - Venue: Proceedings of the 2nd ACM Conference on Advances in Financial Technologies
   - DOI: 10.1145/3419614.3423251
   - Abstract: Analysis of constant function market makers as price oracles
   - Key Contributions: Oracle properties, manipulation resistance

8. **When does the tail wag the dog? Curvature and market making**
   - Authors: Angeris, G., Agrawal, A., Evans, A., Chitra, T., Boyd, S.
   - Year: 2021
   - Venue: arXiv preprint
   - arXiv: 2012.08040
   - Abstract: Analysis of curvature in AMM trading functions
   - Key Contributions: Trading function optimization, liquidity analysis

## High-Frequency Trading and Market Microstructure

### Traditional HFT Research

9. **High-Frequency Trading: A Practical Guide to Algorithmic Strategies and Trading Systems**
   - Author: Aldridge, I.
   - Year: 2013
   - Publisher: John Wiley & Sons
   - ISBN: 978-1-118-34315-9
   - Abstract: Comprehensive guide to HFT strategies and implementation
   - Key Contributions: Latency optimization, market making strategies

10. **The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response**
    - Authors: Budish, E., Cramton, P., Shim, J.
    - Year: 2015
    - Venue: The Quarterly Journal of Economics
    - DOI: 10.1093/qje/qjv027
    - Abstract: Analysis of speed competition in financial markets
    - Key Contributions: Market design solutions, welfare analysis

### Blockchain Performance Optimization

11. **OmniLedger: A Secure, Scale-Out, Decentralized Ledger via Sharding**
    - Authors: Kokoris-Kogias, E., Jovanovic, P., Gasser, L., Gailly, N., Syta, E., Ford, B.
    - Year: 2018
    - Venue: IEEE Symposium on Security and Privacy (SP)
    - DOI: 10.1109/SP.2018.000-5
    - Abstract: Sharding-based approach to blockchain scalability
    - Key Contributions: Sharding protocol, atomic cross-shard transactions

12. **Algorand: Scaling Byzantine Agreements for Cryptocurrencies**
    - Authors: Gilad, Y., Hemo, R., Micali, S., Vlachos, G., Zeldovich, N.
    - Year: 2017
    - Venue: Proceedings of the 26th Symposium on Operating Systems Principles
    - DOI: 10.1145/3132747.3132757
    - Abstract: Scalable Byzantine agreement protocol
    - Key Contributions: Cryptographic sortition, fast finality

## Liquidity and Impermanent Loss

13. **Liquidity Math in Uniswap v3**
    - Authors: Adams, H., Zinsmeister, N., Salem, M., Keefer, R., Robinson, D.
    - Year: 2021
    - Venue: Uniswap Labs Technical Documentation
    - URL: https://uniswap.org/whitepaper-v3.pdf
    - Abstract: Mathematical foundations of concentrated liquidity
    - Key Contributions: Concentrated liquidity, capital efficiency improvements

14. **Impermanent Loss in Uniswap v2**
    - Authors: Pintail
    - Year: 2019
    - Venue: Medium/DeFi Analysis
    - URL: https://pintail.medium.com/understanding-uniswap-returns-cc593f3499ef
    - Abstract: Analysis of impermanent loss for liquidity providers
    - Key Contributions: IL quantification, risk analysis

## Cross-Domain Arbitrage

15. **Cross-chain Arbitrage**
    - Authors: Zamyatin, A., Harz, D., Lind, J., Panayiotou, P., Gervais, A., Knottenbelt, W.
    - Year: 2019
    - Venue: International Conference on Financial Cryptography and Data Security
    - DOI: 10.1007/978-3-030-32101-7_16
    - Abstract: Analysis of arbitrage opportunities across blockchain networks
    - Key Contributions: Cross-chain price discovery, arbitrage strategies

## Smart Contract Security

16. **A Survey of Attacks on Ethereum Smart Contracts (SoK)**
    - Authors: Atzei, N., Bartoletti, M., Cimoli, T.
    - Year: 2017
    - Venue: International Conference on Principles of Security and Trust
    - DOI: 10.1007/978-3-662-54455-6_8
    - Abstract: Comprehensive survey of smart contract vulnerabilities
    - Key Contributions: Vulnerability taxonomy, security analysis

17. **MakerDAO: DeFi's Collateralized Stablecoin System**
    - Authors: MakerDAO Team
    - Year: 2017
    - Venue: MakerDAO Purple Paper
    - URL: https://makerdao.com/purple/
    - Abstract: Design and implementation of DAI stablecoin system
    - Key Contributions: Collateralized debt positions, liquidation mechanisms

## Economic Models and Game Theory

18. **Decentralized Exchange Liquidity Pools**
    - Authors: Evans, A.
    - Year: 2021
    - Venue: Stanford University Technical Report
    - Abstract: Economic analysis of liquidity provision incentives
    - Key Contributions: Incentive alignment, fee distribution models

19. **The Economics of Blockchain Consensus**
    - Authors: Biais, B., Bisiere, C., Bouvard, M., Casamatta, C.
    - Year: 2019
    - Venue: Management Science
    - DOI: 10.1287/mnsc.2018.3200
    - Abstract: Game-theoretic analysis of blockchain consensus mechanisms
    - Key Contributions: Incentive compatibility, equilibrium analysis

## Flash Loans and Composability

20. **DeFi Protocols for Loanable Funds: Interest Rates, Liquidity and Market Efficiency**
    - Authors: Gudgeon, L., Perez, D., Harz, D., Livshits, B., Gervais, A.
    - Year: 2020
    - Venue: Proceedings of the 2nd ACM Conference on Advances in Financial Technologies
    - DOI: 10.1145/3419614.3423254
    - Abstract: Analysis of DeFi lending protocols and their efficiency
    - Key Contributions: Interest rate models, liquidity analysis

## Layer 2 and Scalability Solutions

21. **The Bitcoin Lightning Network: Scalable Off-Chain Instant Payments**
    - Authors: Poon, J., Dryja, T.
    - Year: 2016
    - Venue: Technical White Paper
    - URL: https://lightning.network/lightning-network-paper.pdf
    - Abstract: Payment channel network for Bitcoin scalability
    - Key Contributions: Payment channels, routing algorithms

22. **Plasma: Scalable Autonomous Smart Contracts**
    - Authors: Poon, J., Buterin, V.
    - Year: 2017
    - Venue: Technical White Paper
    - URL: https://plasma.io/plasma.pdf
    - Abstract: Framework for scalable blockchain applications
    - Key Contributions: Child chain architecture, fraud proofs

## Privacy and Anonymity

23. **Tornado Cash: Improving Transaction Privacy on Ethereum**
    - Authors: Tornado Cash Team
    - Year: 2019
    - Venue: Technical Documentation
    - URL: https://tornado.cash/
    - Abstract: Privacy-preserving transaction mixing service
    - Key Contributions: Zero-knowledge proofs, transaction privacy

## Oracles and Price Feeds

24. **Chainlink: A Decentralized Oracle Network**
    - Authors: Ellis, S., Juels, A., Nazarov, S.
    - Year: 2017
    - Venue: Technical White Paper
    - URL: https://link.smartcontract.com/whitepaper
    - Abstract: Decentralized oracle network for smart contracts
    - Key Contributions: Oracle problem solution, decentralized data feeds

## Research Methodology and Tools

25. **Empirical Analysis of DeFi Protocols**
    - Authors: Werner, S.M., Perez, D., Gudgeon, L., Klages-Mundt, A., Harz, D., Knottenbelt, W.J.
    - Year: 2021
    - Venue: arXiv preprint
    - arXiv: 2103.12101
    - Abstract: Systematic analysis of DeFi protocols using on-chain data
    - Key Contributions: Empirical methodology, protocol comparison

## Additional Resources

### Technical Documentation
- Ethereum Yellow Paper: https://ethereum.github.io/yellowpaper/paper.pdf
- Solidity Documentation: https://docs.soliditylang.org/
- Web3.py Documentation: https://web3py.readthedocs.io/

### Data Sources
- DeFi Pulse: https://defipulse.com/
- Dune Analytics: https://dune.xyz/
- The Graph: https://thegraph.com/
- CoinGecko API: https://www.coingecko.com/en/api

### Development Tools
- Hardhat: https://hardhat.org/
- Truffle Suite: https://trufflesuite.com/
- Foundry: https://getfoundry.sh/
- Brownie: https://eth-brownie.readthedocs.io/

### Performance Analysis Tools
- FlameGraph: https://github.com/brendangregg/FlameGraph
- Perf: https://perf.wiki.kernel.org/
- Valgrind: https://valgrind.org/
- Intel VTune: https://software.intel.com/content/www/us/en/develop/tools/vtune-profiler.html

## Conference Venues

### Top-Tier Security/Blockchain Conferences
- IEEE Symposium on Security and Privacy (Oakland)
- USENIX Security Symposium
- ACM Conference on Computer and Communications Security (CCS)
- Network and Distributed System Security Symposium (NDSS)
- International Conference on Financial Cryptography and Data Security (FC)

### Economics and Finance Conferences
- American Finance Association (AFA)
- European Finance Association (EFA)
- Econometric Society Meetings
- American Economic Association (AEA)

### Systems and Performance Conferences
- USENIX Annual Technical Conference (ATC)
- ACM Symposium on Operating Systems Principles (SOSP)
- EuroSys Conference
- ACM SIGMETRICS Conference

---

**Note**: This bibliography is actively maintained and updated as new research emerges in the MEV and DeFi space. For the most current references, please check the respective conference proceedings and preprint servers.

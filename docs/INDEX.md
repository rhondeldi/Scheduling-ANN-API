# Architecture Documentation Index

**Last Updated**: May 16, 2026  
**System Version**: 2.0.0 (Hybrid ANN-GA with 4 Model Stack)

---

## 🎯 Quick Navigation

Choose your documentation based on what you need:

### 📚 **For System Overview**

→ **[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)** ⭐ START HERE

- Complete system architecture
- Block diagrams and data flows
- All 4 ANN models explained
- API endpoints documented
- Integration points detailed
- Performance characteristics
- Deployment structure

### 🚀 **For Getting Started**

→ **[START_HERE.md](START_HERE.md)**

- Step-by-step setup (30-60 minutes)
- Environment configuration
- Data import process
- Model training walkthrough
- Verification steps

### ✅ **For Integration Setup**

→ **[GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md)**

- Current status of integration
- Quick start commands
- Troubleshooting issues
- Verification procedures
- Example payloads

### 📊 **For Data Workflow**

→ **[PROCESS_FLOW.md](PROCESS_FLOW.md)**

- Training data pipeline
- Data generation steps
- Model training phases
- Data flow diagrams
- Synchronization points

### 🔍 **For Implementation Details**

→ **[ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)**

- Neural network architectures
- Feature engineering details
- Training best practices
- Hyperparameter settings
- Optimization strategies

### 📁 **For Data Import**

→ **[DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md)**

- Format requirements
- Import procedures
- Data validation
- Common issues

### 📝 **For Manual Schedules**

→ **[MANUAL_SCHEDULES_GUIDE.md](MANUAL_SCHEDULES_GUIDE.md)**

- JSON format specification
- Manual schedule creation
- Fitness calculation
- Format examples

### 📖 **For Quick Reference**

→ **[../README.md](../README.md)**

- All features at a glance
- Quick command reference
- Common workflows
- Configuration summary

---

## 📋 Documentation Hierarchy

```
START_HERE.md (5-10 min read)
    ↓
SYSTEM_ARCHITECTURE.md (20-30 min read) ← COMPREHENSIVE
    ├─→ GA_ANN_INTEGRATION_WORKING.md (setup)
    ├─→ PROCESS_FLOW.md (training pipeline)
    └─→ ANN_IMPLEMENTATION_GUIDE.md (detailed)

Supporting Docs:
├─→ DATA_IMPORT_GUIDE.md
├─→ MANUAL_SCHEDULES_GUIDE.md
└─→ README.md (quick ref)
```

---

## 🗺️ Architecture Components & Where They're Documented

### Frontend / Client

**Documented In**: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#frontend-client-layer)

- Schedule generation requests
- Parameter configuration
- Results visualization

### Go Backend (Genetic Algorithm)

**Documented In**:

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#go-backend-genetic-algorithm)
- [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md)

Key Files:

- `SchedulePost.go` - Request routing
- `GeneticAlgorithm.go` - GA orchestration
- `ANNClient.go` - Python API communication
- `Crossover.go` - Breeding logic
- `ValidateIndividual.go` - Validation

### Python FastAPI Service

**Documented In**:

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#python-fastapi-service)
- [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)
- [README.md](../README.md)

Code:

- `src/api_service.py` - FastAPI application
- `src/feature_extraction.py` - 48-feature extraction
- `src/models.py` - Keras model definitions
- `scripts/train_*.py` - Training scripts

### 4 ANN Models

**Documented In**:

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#ann-models-python--keras)
- [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md#model-definitions)

Models:

1. **Fitness Predictor** - Rank schedules
2. **Constraint Classifier** - Detect violations
3. **Crossover Recommender** - Guide breeding
4. **Mutation Predictor** - Judge mutations

### Feature Engineering

**Documented In**:

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#feature-extraction-python)
- [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md#feature-extraction)

Features: 48 dimensions across 5 groups

- Temporal Distribution (12)
- Constraint Indicators (12)
- Resource Utilization (7)
- Distribution Quality (9)
- Workload Balance (8)

### Training Data Pipeline

**Documented In**:

- [PROCESS_FLOW.md](PROCESS_FLOW.md)
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#training-data-generation-flow)
- [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md)

Scripts:

- `import_existing_data.py` - Load historical schedules
- `generate_synthetic_variants.py` - Create variants
- `calculate_fitness_for_historical.py` - Add labels
- `train_fitness_predictor.py` - Train models

---

## 🔄 Common Workflows

### "I'm New - Where Do I Start?"

1. Read: [START_HERE.md](START_HERE.md) (5-10 min)
2. Read: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) (20-30 min)
3. Reference: [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md)
4. Run: `python setup.py` then `python src/api_service.py`

### "How Do I Import My Own Schedules?"

→ [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md)
→ [MANUAL_SCHEDULES_GUIDE.md](MANUAL_SCHEDULES_GUIDE.md)

Steps:

1. Prepare JSON file with manual schedules
2. Run `python scripts/import_existing_data.py`
3. Run training scripts
4. Verify with `GET /health`

### "How Does the Hybrid Flow Actually Work?"

→ [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#runtime-prediction-flow-ga-iteration)

Steps:

1. GA creates population
2. Go backend sends batch prediction request
3. Python API extracts features, predicts fitness
4. Go backend ranks, selects, breeds
5. Repeat until convergence

### "What Are the API Endpoints?"

→ [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#api-endpoints)

Endpoints:

- `POST /predict/fitness/batch` - Rank schedules
- `POST /predict/constraints` - Validate schedule
- `POST /recommend/crossover` - Guide breeding
- `POST /predict/mutation` - Judge mutation
- `GET /health` - Service status

### "How Do I Troubleshoot Performance?"

→ [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#performance-characteristics)
→ [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md#troubleshooting--common-issues)

Check:

- Latency targets (should be < 15ms for batch)
- Resource usage (models use ~2.5 GB)
- Use batch requests for efficiency
- Check logs for errors

### "Can I Retrain Models?"

→ [PROCESS_FLOW.md](PROCESS_FLOW.md)
→ [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)

Scripts:

- `train_fitness_predictor.py`
- `train_constraint_classifier.py`
- `train_crossover_recommender.py`
- `train_mutation_predictor.py`

---

## 📊 Document Sizes & Read Times

| Document                                                          | Size   | Read Time | Purpose           |
| ----------------------------------------------------------------- | ------ | --------- | ----------------- |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)                  | ~15 KB | 25-35 min | Complete overview |
| [START_HERE.md](START_HERE.md)                                    | ~8 KB  | 10-15 min | Getting started   |
| [PROCESS_FLOW.md](PROCESS_FLOW.md)                                | ~6 KB  | 10-15 min | Data pipeline     |
| [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)        | ~12 KB | 20-25 min | Implementation    |
| [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md) | ~4 KB  | 5-10 min  | Quick start       |
| [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md)                      | ~3 KB  | 5 min     | Data import       |
| [MANUAL_SCHEDULES_GUIDE.md](MANUAL_SCHEDULES_GUIDE.md)            | ~3 KB  | 5 min     | Schedule format   |
| [README.md](../README.md)                                         | ~10 KB | 10-15 min | Quick reference   |

**Total Documentation**: ~60 KB  
**Total Read Time**: 90-150 minutes

---

## 🎓 Learning Path

### Path 1: Quick Start (30 minutes)

1. [START_HERE.md](START_HERE.md) - Setup (10 min)
2. [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md) - Run it (5 min)
3. [README.md](../README.md) - Reference (15 min)

### Path 2: Full Understanding (2 hours)

1. [START_HERE.md](START_HERE.md) - Context (10 min)
2. [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Overview (30 min)
3. [PROCESS_FLOW.md](PROCESS_FLOW.md) - Pipeline (15 min)
4. [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md) - Details (25 min)
5. [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md) - Integration (10 min)
6. [README.md](../README.md) - Quick ref (10 min)

### Path 3: Implementation Focus (90 minutes)

1. [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Context (30 min)
2. [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md) - Details (25 min)
3. [PROCESS_FLOW.md](PROCESS_FLOW.md) - Pipeline (15 min)
4. [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md) - Data (10 min)
5. [MANUAL_SCHEDULES_GUIDE.md](MANUAL_SCHEDULES_GUIDE.md) - Format (5 min)
6. Hands-on: [START_HERE.md](START_HERE.md) (15 min)

---

## 🔗 Cross-References

### Models Explained In:

- **Fitness Predictor**: [SYSTEM_ARCHITECTURE.md§Model-1](SYSTEM_ARCHITECTURE.md#model-1-fitness-predictor), [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md), [README.md](../README.md)
- **Constraint Classifier**: [SYSTEM_ARCHITECTURE.md§Model-2](SYSTEM_ARCHITECTURE.md#model-2-constraint-classifier), [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)
- **Crossover Recommender**: [SYSTEM_ARCHITECTURE.md§Model-3](SYSTEM_ARCHITECTURE.md#model-3-crossover-recommender), [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)
- **Mutation Predictor**: [SYSTEM_ARCHITECTURE.md§Model-4](SYSTEM_ARCHITECTURE.md#model-4-mutation-predictor), [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)

### Setup Explained In:

- **Environment**: [START_HERE.md](START_HERE.md), [README.md](../README.md)
- **Models & Scalers**: [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md), [START_HERE.md](START_HERE.md)
- **API Service**: [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md), [README.md](../README.md)

### Data Explained In:

- **Formats**: [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md), [MANUAL_SCHEDULES_GUIDE.md](MANUAL_SCHEDULES_GUIDE.md)
- **Pipeline**: [PROCESS_FLOW.md](PROCESS_FLOW.md), [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
- **Training**: [START_HERE.md](START_HERE.md), [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)

### Integration Explained In:

- **Architecture**: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
- **Setup**: [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md)
- **Testing**: [README.md](../README.md)

---

## 💡 Tips for Using This Documentation

1. **Skim First**: Read the overview sections first to get context
2. **Reference Later**: Use detailed sections when implementing
3. **Keep Bookmarks**: Save links to commonly used sections
4. **Check Examples**: All guides include example code/JSON
5. **Follow Learning Path**: Use the suggested path for your use case
6. **Use Search**: Search for keywords across all docs (Ctrl+F)

---

## 📞 Quick Help

| Question                    | Answer Location                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------- |
| How do I start?             | [START_HERE.md](START_HERE.md)                                                                  |
| What is the architecture?   | [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)                                                |
| How do I import data?       | [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md)                                                    |
| What are the models?        | [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)                                      |
| What's the training flow?   | [PROCESS_FLOW.md](PROCESS_FLOW.md)                                                              |
| How do I integrate with Go? | [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md)                               |
| What commands do I run?     | [README.md](../README.md)                                                                       |
| What format for schedules?  | [MANUAL_SCHEDULES_GUIDE.md](MANUAL_SCHEDULES_GUIDE.md)                                          |
| What are the API endpoints? | [SYSTEM_ARCHITECTURE.md§API-Endpoints](SYSTEM_ARCHITECTURE.md#api-endpoints)                    |
| How do I troubleshoot?      | [SYSTEM_ARCHITECTURE.md§Troubleshooting](SYSTEM_ARCHITECTURE.md#troubleshooting--common-issues) |

---

## 📝 Document Maintenance

**Last Updated**: May 16, 2026  
**Version**: 2.0.0  
**Status**: ✅ Complete and current

**Recent Changes**:

- ✅ Created comprehensive SYSTEM_ARCHITECTURE.md
- ✅ Updated visualize_system.py with references
- ✅ Created this INDEX.md for navigation
- ✅ Added API endpoint specifications
- ✅ Added performance characteristics
- ✅ Added troubleshooting guide

---

**Next Steps**:

1. Choose your learning path above
2. Start with the recommended entry document
3. Reference other docs as needed
4. Run the commands in START_HERE.md or GA_ANN_INTEGRATION_WORKING.md
5. Refer back to this INDEX.md when you need something specific

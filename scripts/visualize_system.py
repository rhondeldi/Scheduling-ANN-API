"""
Visual demonstration of the Hybrid ANN-Assisted Genetic Algorithm system.
Run this script to print the backend + ANN architecture and flow.
"""

from textwrap import dedent


def print_architecture():
    """Print the hybrid system architecture."""

    print(
        dedent(
            """
            ================================================================================
            HYBRID ANN-ASSISTED GENETIC ALGORITHM SCHEDULER
            ================================================================================

            SYSTEM ARCHITECTURE
            -------------------------------------------------------------------------------

                [Frontend / Client]
                    - Request schedule generation
                    - Display schedules and progress
                    - Show results and validation status
                             |
                             | HTTP
                             v
                [Go Backend]
                    - Route requests
                    - Run the Genetic Algorithm
                    - Validate and repair schedules
                    - Bridge requests to the ANN API
                             |
               +-------------+-------------+-------------+-------------+
               |             |             |             |             |
               | POST        | POST        | POST        | POST        |
               | /predict/   | /predict/   | /recommend/| /predict/   |
               | fitness/    | constraints | crossover  | mutation    |
               | batch       |             |            |             |
               v             v             v             v             v
                [Python ANN API - FastAPI]
                    - Loads trained models at startup
                    - Serves ANN predictions for the GA
                    - Returns structured JSON responses

                    1. Fitness Predictor
                       - Ranks schedules in the GA population
                       - Endpoint: /predict/fitness/batch

                    2. Constraint Classifier
                       - Flags risky or invalid schedules
                       - Endpoint: /predict/constraints

                    3. Crossover Recommender
                       - Suggests crossover split points
                       - Endpoint: /recommend/crossover

                    4. Mutation Predictor
                       - Judges proposed mutations
                       - Endpoint: /predict/mutation

            -------------------------------------------------------------------------------
            HOW THE HYBRID FLOW WORKS
            -------------------------------------------------------------------------------

                1. The backend prepares or repairs the base university schedule.
                2. The GA creates the genesis population.
                3. The fitness model scores schedules for ranking.
                4. The crossover model guides offspring creation.
                5. The constraint model catches invalid schedules early.
                6. The mutation model helps decide whether to accept a mutation.
                7. The backend validates the final schedule before saving.

            -------------------------------------------------------------------------------
            BACKEND FILES THAT MATTER
            -------------------------------------------------------------------------------

                - Routes/RoutesV1/SchedulePost.go
                  Enables ANN mode and passes the ANN client into the GA.

                - GeneticAlgorithm/GeneticAlgorithm.go
                  Orchestrates population ranking, crossover, advisory checks,
                  mutation flow, and final validation.

                - GeneticAlgorithm/Crossover.go
                  Uses crossover recommendations and repairs offspring.

                - GeneticAlgorithm/ANNClient.go
                  Sends requests from Go to the Python ANN API.

                - GeneticAlgorithm/ValidateIndividual.go
                  Reports missing subjects, overlaps, and horizontal validation errors.

            -------------------------------------------------------------------------------
            ANN API FILE THAT MATTERS
            -------------------------------------------------------------------------------

                - src/api_service.py
                  Loads the 4 models and exposes the matching endpoints.

            ================================================================================
            """
        ).strip()
    )


def print_model_roles():
    """Print the four ANN model roles in the hybrid system."""

    print(
        dedent(
            """
            MODEL ROLES
            -------------------------------------------------------------------------------

            Fitness Predictor
                Input:    Full schedule batch
                Output:   Fitness score
                Used for: Population ranking and selection pressure

            Constraint Classifier
                Input:    One schedule
                Output:   Constraint violation scores
                Used for:  Rejecting or repairing risky schedules

            Crossover Recommender
                Input:    Parent schedules + parent fitness values
                Output:   Suggested crossover points
                Used for:  Better offspring construction

            Mutation Predictor
                Input:    Current schedule + proposed mutation
                Output:   improve / neutral / worsen
                Used for:  Deciding whether a mutation should stay
            """
        ).strip()
    )


def print_feature_extraction_example():
    """Show how feature extraction supports all 4 models."""

    print(
        dedent(
            """
            FEATURE EXTRACTION EXAMPLE
            -------------------------------------------------------------------------------

            Sample schedule data:
                6 days x 24 time slots x 3 attributes

            Raw schedule
                [subject_id, instructor_id, room_id]
                [0, 0, 0] means an empty slot

            Feature vector produced by the ANN pipeline:
                - day-hour counts
                - lunch / late-class indicators
                - room and instructor usage
                - schedule distribution statistics

            That same schedule representation is reused by:
                - Fitness model      -> score the schedule
                - Constraint model   -> detect violations
                - Crossover model    -> choose split points
                - Mutation model     -> evaluate proposed changes

            Final outcome:
                The GA makes better decisions with ANN guidance instead of
                relying only on random crossover and mutation.
            """
        ).strip()
    )


if __name__ == "__main__":
    print_architecture()
    print()
    print_model_roles()
    print()
    print_feature_extraction_example()
    print()
    print("=" * 79)
    print("Next step: start the ANN API, confirm /health shows all 4 models loaded,")
    print("and then run schedule generation so the backend can use the hybrid flow.")
    print("=" * 79)
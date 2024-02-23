# Test with saved weights
    # print("TESTING PERFORMANCE")
    # rl_agent_submission.load()
    # rl_agent_submission.set_training_mode(False)
    # arena = LemonadeArena(
    #     num_rounds=1000,
    #     timeout=1,
    #     players=[
    #         rl_agent_submission,
    #         StickAgent("Bug1"),
    #         ReserveAgent("Bug2"),
    #         DecrementAgent("Bug3"),
    #         IncrementAgent("Bug4")
    #     ]
    # )
    # # NOTE: FEEL FREE TO EDIT THE AGENTS HERE TO TEST AGAINST A DIFFERENT DISTRIBUTION OF AGENTS. A COUPLE OF EXAMPLE AGENTS
    # #       TO TEST AGAINST ARE IMPORTED FOR YOU. 
    # arena.run()
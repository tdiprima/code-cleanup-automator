# This script demonstrates replacing a long if-elif chain with a dispatch
# dictionary for handling events. It includes example handler functions
# and a main function to simulate event processing.

def handle_start():
    print("Starting the process...")


def handle_stop():
    print("Stopping the process...")


def handle_pause():
    print("Pausing the process...")


def handle_unknown():
    print("Unknown event encountered.")


# Dispatch table
event_actions = {
    "start": handle_start, 
    "stop": handle_stop, 
    "pause": handle_pause
}


def process_event(event):
    action = event_actions.get(event, handle_unknown)
    action()


# Example usage
if __name__ == "__main__":
    process_event("start")
    process_event("resume")  # This will trigger unknown

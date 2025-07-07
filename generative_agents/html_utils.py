import json
import os, re
import base64

# Enhanced header with dynamic CSS for multiple agents
header = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .agent-info {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chat {
            display: flex;
            flex-direction: column;
        }
        .message {
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 15px;
            font-size: 16px;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            position: relative;
        }
        .speaker-name {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 5px;
            opacity: 0.8;
        }
        .date {
            background-color: #ffb6c1;
            align-self: center;
            max-width: 400px;
            text-align: center;
            font-weight: bold;
            margin: 20px 0;
        }
        .message img {
            max-width: 100%;
            height: auto;
            margin-top: 10px;
            border-radius: 5px;
        }
        .events {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            font-size: 14px;
        }
        /* Dynamic agent colors - will be populated by JavaScript */
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Agent Conversation</h1>
        <div id="agent-info-section">
            <!-- Agent info will be inserted here -->
        </div>
        <div class="chat" id="chat-container">
"""

# Agent color palette
AGENT_COLORS = [
    {"bg": "#e2f0cb", "align": "flex-start"},  # Light green
    {"bg": "#b2ebf2", "align": "flex-end"},    # Light blue  
    {"bg": "#ffcccb", "align": "flex-start"},  # Light red
    {"bg": "#e1bee7", "align": "flex-end"},    # Light purple
    {"bg": "#fff9c4", "align": "flex-start"},  # Light yellow
    {"bg": "#ffcc80", "align": "flex-end"},    # Light orange
    {"bg": "#c8e6c9", "align": "flex-start"},  # Light mint
    {"bg": "#f8bbd9", "align": "flex-end"},    # Light pink
]

def get_agent_color_css(agents):
    """Generate CSS for agent-specific styling"""
    css = ""
    for i, agent in enumerate(agents):
        color_info = AGENT_COLORS[i % len(AGENT_COLORS)]
        agent_class = f"agent-{i}"
        css += f"""
        .{agent_class} {{
            background-color: {color_info['bg']};
            align-self: {color_info['align']};
        }}
        """
    return css

def get_speaker_info(speaker, use_events=False):
    """Generate HTML for speaker information"""
    output = ""
    output += "<b>Name</b>: " + speaker["name"] + '<br>'
    if 'persona_summary' in speaker:
        output += "<b>Persona</b>: " + speaker["persona_summary"] + '<br>'
    return output

def get_session_events(events):
    """Generate HTML for session events"""
    output = '<div class="events"><b>Events</b><br>'
    for e in events:
        try:
            event_text = e.get('sub-event', e.get('sub_event', 'Unknown event'))
            date = e.get('date', 'Unknown date')
            output += f'<b>{date}</b>: {event_text}<br>'
        except (KeyError, TypeError):
            output += f'Event: {str(e)}<br>'
    output += '</div>'
    return output

def img2base64(image_file_path):
    """Convert image to base64 string"""
    try:
        with open(image_file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_file_path}: {e}")
        return ""

def convert_to_chat_html_multi(agents, outfile="", use_events=False, img_dir=None):
    """
    Convert multi-agent conversation to HTML format.
    
    Args:
        agents: List of agent dictionaries
        outfile: Output HTML file path
        use_events: Whether to include events in the output
        img_dir: Directory containing images
    """
    if not agents:
        print("No agents provided for HTML conversion")
        return
    
    # Create agent name to index mapping
    agent_name_to_idx = {agent['name']: i for i, agent in enumerate(agents)}
    
    body = header
    
    # Add dynamic CSS for agent colors
    body += "<style>" + get_agent_color_css(agents) + "</style>"
    
    # Add agent information section
    for i, agent in enumerate(agents):
        agent_color = AGENT_COLORS[i % len(AGENT_COLORS)]['bg']
        body += f"""
        <div class="agent-info" style="border-left: 5px solid {agent_color};">
            <h3>Agent {i+1}: {agent['name']}</h3>
            {get_speaker_info(agent, use_events=use_events)}
        </div>
        """
    
    body += "</div><div class=\"chat\" id=\"chat-container\">"
    
    # Process sessions
    for session_num in range(1, 50):
        session_key = f'session_{session_num}'
        
        # Check if this session exists in any agent
        if not any(session_key in agent for agent in agents):
            break
        
        # Get session data from the first agent that has it
        main_agent = next(agent for agent in agents if session_key in agent)
        
        # Add session header
        if f'session_{session_num}_date_time' in main_agent:
            date_time_string = main_agent[f'session_{session_num}_date_time']
        elif f'session_{session_num}_date' in main_agent:
            date_time_string = main_agent[f'session_{session_num}_date']
        else:
            date_time_string = f"Session {session_num}"
        
        body += f"""
        <div class="message date">
            <p>Session {session_num} [{date_time_string}]</p>
        </div>
        """
        
        # Add events if available and requested
        if use_events:
            events_key = f'events_session_{session_num}'
            for i, agent in enumerate(agents):
                if events_key in agent and agent[events_key]:
                    agent_class = f"agent-{i}"
                    body += f"""
                    <div class="message {agent_class}">
                        <div class="speaker-name">{agent['name']}'s Events</div>
                        {get_session_events(agent[events_key])}
                    </div>
                    """
        
        # Add conversation dialogs
        if session_key in main_agent:
            for dialog in main_agent[session_key]:
                speaker_name = dialog.get("speaker", "Unknown")
                text = dialog.get("clean_text", dialog.get("text", ""))
                
                # Find agent index for this speaker
                agent_idx = agent_name_to_idx.get(speaker_name, 0)
                agent_class = f"agent-{agent_idx}"
                
                # Handle images
                if "img_url" in dialog:
                    try:
                        url = dialog["img_url"]
                        if isinstance(url, list):
                            url = url[0]
                        caption = dialog.get('caption', 'Image')
                        
                        body += f"""
                        <div class="message {agent_class}">
                            <div class="speaker-name">{speaker_name}</div>
                            <p>{text}</p>
                            <img src="{url}" alt="Shared image" style="width:300px;">
                            <p><em>{caption}</em></p>
                        </div>
                        """
                    except Exception as e:
                        print(f"Error processing image for dialog: {e}")
                        body += f"""
                        <div class="message {agent_class}">
                            <div class="speaker-name">{speaker_name}</div>
                            <p>{text}</p>
                        </div>
                        """
                else:
                    body += f"""
                    <div class="message {agent_class}">
                        <div class="speaker-name">{speaker_name}</div>
                        <p>{text}</p>
                    </div>
                    """
    
    # Close HTML
    body += """
        </div>
    </div>
    
    <script>
        // Add agent filtering functionality
        function createAgentFilters() {
            const agentInfo = document.getElementById('agent-info-section');
            const filterDiv = document.createElement('div');
            filterDiv.style.cssText = 'margin: 20px 0; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);';
            filterDiv.innerHTML = '<h3>Filter by Agent:</h3>';
            
            // Add "Show All" button
            const showAllBtn = document.createElement('button');
            showAllBtn.textContent = 'Show All';
            showAllBtn.style.cssText = 'margin: 5px; padding: 8px 15px; border: none; border-radius: 5px; background: #007bff; color: white; cursor: pointer;';
            showAllBtn.onclick = () => showAllAgents();
            filterDiv.appendChild(showAllBtn);
            
            // Add individual agent filter buttons
            const agentClasses = Array.from(document.querySelectorAll('[class*="agent-"]'))
                .map(el => el.className.match(/agent-\\d+/)?.[0])
                .filter((cls, index, arr) => cls && arr.indexOf(cls) === index);
            
            agentClasses.forEach((agentClass, index) => {
                const agentName = document.querySelector(`.${agentClass} .speaker-name`)?.textContent || `Agent ${index + 1}`;
                const btn = document.createElement('button');
                btn.textContent = agentName;
                btn.style.cssText = `margin: 5px; padding: 8px 15px; border: none; border-radius: 5px; background: ${getAgentColor(index)}; cursor: pointer;`;
                btn.onclick = () => filterByAgent(agentClass);
                filterDiv.appendChild(btn);
            });
            
            agentInfo.appendChild(filterDiv);
        }
        
        function getAgentColor(index) {
            const colors = ['#e2f0cb', '#b2ebf2', '#ffcccb', '#e1bee7', '#fff9c4', '#ffcc80', '#c8e6c9', '#f8bbd9'];
            return colors[index % colors.length];
        }
        
        function showAllAgents() {
            document.querySelectorAll('.message').forEach(msg => {
                msg.style.display = 'flex';
            });
        }
        
        function filterByAgent(agentClass) {
            document.querySelectorAll('.message').forEach(msg => {
                if (msg.classList.contains(agentClass) || msg.classList.contains('date')) {
                    msg.style.display = 'flex';
                } else {
                    msg.style.display = 'none';
                }
            });
        }
        
        // Add session navigation
        function createSessionNavigation() {
            const container = document.querySelector('.container');
            const navDiv = document.createElement('div');
            navDiv.style.cssText = 'position: sticky; top: 20px; z-index: 100; margin: 20px 0; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);';
            navDiv.innerHTML = '<h3>Jump to Session:</h3>';
            
            const sessions = document.querySelectorAll('.message.date');
            sessions.forEach((session, index) => {
                const btn = document.createElement('button');
                btn.textContent = `Session ${index + 1}`;
                btn.style.cssText = 'margin: 5px; padding: 8px 15px; border: none; border-radius: 5px; background: #28a745; color: white; cursor: pointer;';
                btn.onclick = () => session.scrollIntoView({ behavior: 'smooth', block: 'start' });
                navDiv.appendChild(btn);
            });
            
            container.insertBefore(navDiv, container.firstChild);
        }
        
        // Initialize interactive features
        document.addEventListener('DOMContentLoaded', function() {
            createAgentFilters();
            createSessionNavigation();
            
            // Add smooth scrolling for better UX
            document.documentElement.style.scrollBehavior = 'smooth';
        });
    </script>
</body>
</html>
"""
    
    # Write to file
    try:
        with open(outfile, 'w', encoding='utf-8') as fhtml:
            fhtml.write(body)
        print(f"Multi-agent HTML conversation saved to: {outfile}")
    except Exception as e:
        print(f"Error writing HTML file: {e}")

# Legacy function for backward compatibility
def convert_to_chat_html(speaker_1, speaker_2, outfile="", use_events=False, img_dir=None):
    """
    Legacy function for two-agent compatibility.
    Redirects to the new multi-agent function.
    """
    agents = [speaker_1]
    if speaker_2['name'] != speaker_1['name']:
        agents.append(speaker_2)
    
    convert_to_chat_html_multi(agents, outfile, use_events, img_dir)


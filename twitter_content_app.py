# twitter_content_app.py

import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.groq import Groq
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.firecrawl import FirecrawlTools
from agno.tools.reasoning import ReasoningTools

#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
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_JdZ")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "fc-6")

# ============== PYDANTIC MODELS ==============

class Tweet(BaseModel):
    content: str = Field(..., description="Tweet content")
    character_count: int = Field(..., description="Character count")
    hashtags: List[str] = Field(default=[], description="Hashtags for this tweet")

class TweetOutput(BaseModel):
    tweets: List[Tweet] = Field(..., description="List of generated tweets")
    is_thread: bool = Field(default=False, description="Whether this is a thread")
    topic_summary: str = Field(..., description="Brief summary of researched topic")

class CompetitorInsight(BaseModel):
    competitor_name: str
    content_strategy: str
    hashtag_patterns: List[str]
    posting_frequency: str
    key_themes: List[str]

# ============== AGENT DEFINITIONS ==============

def create_agents(groq_api_key: str, firecrawl_api_key: Optional[str] = None):
    """Create all agents with provided API keys"""
    
    model = Groq(
        id="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=groq_api_key
    )
    
    # 1. Web Search Agent (Base Research)
    web_search_agent = Agent(
        name="Web Search Agent",
        model=model,
        tools=[DuckDuckGoTools()],
        instructions=[
            "Search the web for current information on the given topic",
            "Find trending discussions and recent news",
            "Identify key talking points and angles",
        ],
        markdown=True,
    )
    
    # 2. Deep Research Agent
    deep_research_agent = Agent(
        name="Deep Research Agent",
        model=model,
        tools=[DuckDuckGoTools()],
        #tools=[DuckDuckGoTools(), ReasoningTools(add_instructions=True)],
        instructions=[
            "Conduct thorough research on the topic",
            "Cross-reference multiple sources for accuracy",
            "Think step-by-step to analyze and synthesize findings",
            "Identify unique angles and insights",
            "Document key statistics and facts with sources",
        ],
        markdown=True,
    )
    
    # 3. Competitor Analysis Agent
    competitor_tools = [DuckDuckGoTools()]
    if firecrawl_api_key:
        competitor_tools.append(
            FirecrawlTools(
                api_key=firecrawl_api_key,
                enable_scrape=True,
                enable_crawl=True,
                enable_search=True,
                limit=3,# why the limit what does it means?
            )
        )
    
    competitor_agent = Agent(
        name="Competitor Analyst",
        model=model,
        tools=competitor_tools,
        instructions=[
            "Analyze competitor Twitter/X accounts and content strategies",
            "Identify successful hashtag patterns",
            "Note engagement patterns and content themes",
            "Scrape competitor websites for content insights if URL provided",
            "Provide actionable competitive intelligence",
        ],
        markdown=True,
    )
    
    # 4. Content Writer Agent
    content_writer = Agent(
        name="Tweet Writer",
        model=model,
        instructions=[
            "Create engaging, viral-worthy tweets based on research",
            "Generate smart, relevant hashtags (3-5 per tweet)",
            "Ensure each tweet is self-contained but works as a thread",
            "Use hooks, questions, and calls-to-action",
            "Match the tone to the topic (professional, casual, etc.)",
        ],
        markdown=True,
    )
    
    return web_search_agent, deep_research_agent, competitor_agent, content_writer

# ============== STREAMLIT APP ==============

def main():
    st.set_page_config(
        page_title="Twitter Content Generator",
        page_icon="🐦",
        layout="wide"
    )
    
    st.title("🐦 AI Twitter Content Generator")
    st.markdown("Generate research-backed tweets with smart hashtags")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Keys
        st.subheader("API Keys")
        groq_api_key = st.text_input(
            "Groq API Key",
            value=" l",
            type="password"
        )
        firecrawl_api_key = st.text_input(
            "Firecrawl API Key (Optional)",
            type="password",
            help="Required for competitor website scraping"
        )
        
        st.divider()
        
        # Account Type
        st.subheader("Account Settings")
        account_type = st.radio(
            "Account Type",
            ["Free (1000 chars)", "Paid (4000 chars)"],
            help="Select your Twitter/X account type"
        )
        char_limit = 1000 if "Free" in account_type else 4000
        st.info(f"Character limit: {char_limit}")
        
        # it will create 5 Tweets or tweet chains
        num_tweets = st.slider(
            "Number of Tweets",
            min_value=1,
            max_value=10,
            value=5,
            help="How many tweets to generate"
        )
        
        # Tweet Type
        tweet_type = st.radio(
            "Tweet Type",
            ["Individual Tweets", "Tweet Thread"],
        )
        
        st.divider()
        
        # Features Toggle
        st.subheader("Features")
        use_deep_research = st.checkbox("Deep Research", value=True)
        use_competitor_analysis = st.checkbox("Competitor Analysis", value=False)
        generate_hashtags = st.checkbox("Smart Hashtags", value=True)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Topic Input
        topic = st.text_area(
            "Topic for Research",
            placeholder="Enter your topic (e.g., 'AI trends in 2024', 'sustainable fashion')",
            height=100
        )
        
        # Custom System Prompt
        with st.expander(" Edit System Prompt (Optional)"):
            default_prompt = """You are a social media expert creating viral Twitter content.
Your tweets should be:
- Engaging and shareable
- Informative yet concise
- Include relevant hashtags
- Have strong hooks and CTAs"""
            
            custom_prompt = st.text_area( # Default system prompt (editable) used inside tweet writer pompt
                "System Prompt",
                value=default_prompt,
                height=150
            )
        
        # Competitor URLs (if enabled)
        competitor_urls = []
        if use_competitor_analysis:
            with st.expander(" Competitor Analysis"):
                competitor_input = st.text_area(
                    "Competitor URLs/Accounts (one per line)",
                    placeholder="https://competitor1.com\n@competitor_twitter",
                    height=100
                )
                if competitor_input:
                    competitor_urls = [url.strip() for url in competitor_input.split("\n") if url.strip()]
        
        # Generate Button
        generate_btn = st.button(" Generate Tweets", type="primary", use_container_width=True)
    
    with col2:
        st.header(" Output")
        
        if generate_btn and topic and groq_api_key:
            with st.spinner(" Generating content..."):
                try:
                    # Create agents
                    web_agent, research_agent, competitor_agent, writer_agent = create_agents(
                        groq_api_key, 
                        firecrawl_api_key if firecrawl_api_key else None
                    )
                    
                    results = {}
                    
                    # Step 1: Web Search
                    with st.status("🔍 Searching the web...", expanded=True) as status:
                        search_response = web_agent.run(f"Search for latest information about: {topic}")
                        results["web_search"] = search_response.content
                        st.write("✅ Web search complete")
                        status.update(label="Web search complete!", state="complete")
                    
                    # Step 2: Deep Research (if enabled)
                    if use_deep_research:
                        with st.status("🧠 Conducting deep research...", expanded=True) as status:
                            research_prompt = f"""
                            Topic: {topic}
                            
                            Previous findings: {results.get('web_search', '')}
                            
                            Conduct deep research and provide:
                            1. Key facts and statistics
                            2. Trending angles
                            3. Unique insights
                            4. Potential controversy or debate points
                            """
                            research_response = research_agent.run(research_prompt)
                            results["deep_research"] = research_response.content
                            st.write("✅ Deep research complete")
                            status.update(label="Deep research complete!", state="complete")
                    
                    # Step 3: Competitor Analysis (if enabled)
                    if use_competitor_analysis and competitor_urls:
                        with st.status("📊 Analyzing competitors...", expanded=True) as status:
                            competitor_prompt = f"""
                            Analyze these competitors for topic '{topic}':
                            {chr(10).join(competitor_urls)}
                            
                            Identify:
                            1. Their content strategy
                            2. Popular hashtags they use
                            3. Engagement patterns
                            4. Content themes
                            """
                            competitor_response = competitor_agent.run(competitor_prompt)
                            results["competitor_analysis"] = competitor_response.content
                            st.write("✅ Competitor analysis complete")
                            status.update(label="Competitor analysis complete!", state="complete")
                    
                    # Step 4: Generate Tweets
                    with st.status("✍️ Writing tweets...", expanded=True) as status:
                        tweet_prompt = f"""
                        {custom_prompt}
                        
                        Topic: {topic}
                        IMPORTANT REQUIREMENTS:
                        - MINIMUM character count per tweet: {int(char_limit * 0.7)} characters
                        - MAXIMUM character count per tweet: {char_limit} characters
                        - Each tweet MUST be at least {int(char_limit * 0.7)} characters long
                        - You have room for {char_limit} characters, USE IT for detailed content
                        Number of Tweets: {num_tweets}
                        Type: {"Thread (connected tweets)" if tweet_type == "Tweet Thread" else "Individual standalone tweets"}
                        Generate Hashtags: {generate_hashtags}
                        
                        Research Findings:
                        {results.get('web_search', '')}
                        
                        {results.get('deep_research', '') if use_deep_research else ''}
                        
                        {results.get('competitor_analysis', '') if use_competitor_analysis else ''}
                        
                        Generate {num_tweets} tweets. For each tweet:
                        1. Write engaging content under {char_limit} characters
                        2. Include 3-5 relevant hashtags
                        3. Show character count
                        
                        Format each tweet as:
                        Tweet [number]:
                        [content]
                        Hashtags: #tag1 #tag2 #tag3
                        Characters: [count]
                        """
                        
                        tweet_response = writer_agent.run(tweet_prompt)
                        results["tweets"] = tweet_response.content
                        status.update(label="Tweets generated!", state="complete")
                    
                    # Display Results
                    st.success("✅ Content generated successfully!")
                    
                    # Show Research Summary
                    if use_deep_research:
                        with st.expander("📚 Research Summary", expanded=False):
                            st.markdown(results.get("deep_research", ""))
                    
                    if use_competitor_analysis and competitor_urls:
                        with st.expander("📊 Competitor Insights", expanded=False):
                            st.markdown(results.get("competitor_analysis", ""))
                    
                    # Show Generated Tweets
                    st.subheader("🐦 Generated Tweets")
                    st.markdown(results.get("tweets", ""))
                    
                    # Copy functionality
                    st.download_button(
                        label="📋 Download Tweets",
                        data=results.get("tweets", ""),
                        file_name="generated_tweets.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
        
        elif generate_btn and not topic:
            st.warning("Please enter a topic to research")
        elif generate_btn and not groq_api_key:
            st.warning("Please enter your Groq API key")

if __name__ == "__main__":
    main()

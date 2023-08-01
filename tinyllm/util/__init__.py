
import datetime as dt

# Create Story nodes
story1 = Story(timestamp=dt.datetime(2023, 7, 13, 1, 31), title="Analysis: The Impact of Bitcoin ETF Approval and Ethereum Merge",
               summary="Cryptocurrency experts debate the potential ramifications of the proposed Bitcoin ETF and the upcoming Ethereum Merge. Optimists point to increased institutional acceptance, while skeptics express concern over regulatory challenges and technical obstacles.")
story2 = Story(timestamp=dt.datetime(2023, 7, 12, 1, 31), title="Mike Novogratz on BTC : 'it's going to the moon'",
               summary="Mike Novogratz, a renowned former hedge fund manager of Galaxy Capital and prominent cryptocurrency advocate, has recently made a bullish statement about Bitcoin's future. In a televised interview, Novogratz asserted that Bitcoin (BTC) is 'going to the moon,' signifying his strong belief in the cryptocurrency's potential for substantial growth. Drawing on his experience and analysis of market trends, he laid out a compelling case for why investors and the general public should be optimistic about Bitcoin's prospects.")

# Create Person nodes
person1 = Person(name="Alexander Victor", type="journalist")
person2 = Person(name="Mike Novogratz", type="businessman")

# Create Organization nodes
organization1 = Organization(name="Cointelegraph", type="media outlet")
organization2 = Organization(name="Galaxy Capital", type="investment firm")

# Create Cryptoasset nodes
cryptoasset1 = Cryptoasset(symbol="BTC", name="Bitcoin", type="cryptocurrency")

# Create Post nodes
post1 = Post(sup_id=92, author=person1)
post2 = Post(sup_id=93, author=person2)

# Create Topic nodes
topic1 = Topic(name="Bitcoin ETF")
topic2 = Topic(name="Ethereum Merge")

# Create Tag nodes
tag1 = Tag(name="analysis")
tag2 = Tag(name="opinion")

# Create relationships
story1.generated.connect(post1)
story2.generated.connect(post2)

story1.is_type.connect(tag1)
story2.is_type.connect(tag2)

story1.mentions_topic.connect(topic1)
story1.mentions_topic.connect(topic2)
story2.mentions_topic.connect(topic1)

story1.mentions_cryptoasset.connect(cryptoasset1)
story2.mentions_cryptoasset.connect(cryptoasset1)

story1.mentions_person.connect(person1)
story2.mentions_person.connect(person2)

story1.mentions_organization.connect(organization1)
story2.mentions_organization.connect(organization2)

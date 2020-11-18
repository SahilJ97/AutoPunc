IWSLT 2012 evaluation campaign: test sets for MT track

# Copyright: TED Conference LLC
# License: Creative Commons Attribution-NonCommercial-NoDerivs 3.0

This directory includes the evaluation sets of the MT track for one of
the following language pairs:

- English to French

- {Arabic, German, Dutch, Polish, Portoguese-Brazil, Romanian,
   Russian, Slovenian, Turkish, Chinese} to English

For each pair x-y of languages, the xml files

IWSLT12.TED.MT.tst2011.x-y.x.xml
IWSLT12.TED.MT.tst2012.x-y.x.xml

are provided, with the exception of Slovenian-English pair for which
only one evaluation set is available:

EU-BRIDGE.TED.MT.tst2012.sl-en.sl.xml

The evaluation sets of pairs towards English mostly include the same
TED talks - again with the exception of Slovenian-English - which are
disjoint from those included in the English-French sets.

Each talk defines a single document, for which the following tags are
provided:

<url>: the url of the page with the text
<description>: a brief description of the talk
<keywords>: keywords of the talk
<talkid>: a numeric identifier of the talk
<title>: the title of the talk

Text is segmented in sentences. Segments can include more than one
sentence.

UTF8 encoding is used.  The simplified set of characters is employed
for Chinese text. No segmentation/preprocessing has been applied to
Arabic nor to Chinese.
